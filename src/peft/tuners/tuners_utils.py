# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import copy
import os
import re
import textwrap
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext
from typing import Any, Optional, Union, overload

import torch
from accelerate.hooks import AlignDevicesHook
from accelerate.utils import named_module_tensors, offload_state_dict
from torch import nn
from transformers import PreTrainedModel
from transformers.pytorch_utils import Conv1D

from peft.mapping import PEFT_TYPE_TO_PREFIX_MAPPING
from peft.utils import INCLUDE_LINEAR_LAYERS_SHORTHAND
from peft.utils.constants import (
    DUMMY_MODEL_CONFIG,
    DUMMY_TARGET_MODULES,
    EMBEDDING_LAYER_NAMES,
    MIN_TARGET_MODULES_FOR_OPTIMIZATION,
    SEQ_CLS_HEAD_NAMES,
)
from peft.utils.integrations import init_empty_weights
from peft.utils.other import AuxiliaryTrainingWrapper, match_target_against_key, set_additional_trainable_modules
from peft.utils.peft_types import PeftType, TaskType

from ..config import PeftConfig
from ..utils import _get_submodules
from ._buffer_dict import BufferDict


@contextmanager
def onload_layer(layer):
    r"""
    A utility for modifying a module containing one or more tuners and a base layer, any of which are offloaded to the
    CPU or disk. Moves a module's sub-modules to the execution device before some action is performed, after that the
    base layer state dictionary is re-assigned (if that layer was offloaded to the disk) and finally the parameters are
    offloaded.

    If the module has no offloaded sub-modules, this function does nothing.

    Args:
        layer ('torch.nn.Module'):
            layer with tuners to be merged
    """

    offloaded_modules = []
    for name, module in layer.named_modules():
        if name in ["", "base_layer"]:
            continue
        if hasattr(module, "_hf_hook") and isinstance(module._hf_hook, AlignDevicesHook) and module._hf_hook.offload:
            module._hf_hook.pre_forward(module)
            offloaded_modules.append(module)

    base_layer_offload = False
    if hasattr(layer, "base_layer") and (
        hasattr(layer.base_layer, "_hf_hook")
        and isinstance(layer.base_layer._hf_hook, AlignDevicesHook)
        and layer.base_layer._hf_hook.offload
    ):
        # check if the base layer is disk-offloaded (must contain a 'dataset' and an offload index)
        if torch.device("meta") in layer.base_layer._hf_hook.original_devices.values() and hasattr(
            layer.base_layer._hf_hook.weights_map, "dataset"
        ):
            # find the disk-offload index (maps modules to safetensors) from the `dataset` (OffloadedWeightsLoader object)
            index = layer.base_layer._hf_hook.weights_map.dataset.index
            module_name = list(dict(layer.base_layer._hf_hook.weights_map.dataset).keys())[0]  # any module will do
            file_name = index[module_name]["safetensors_file"]
            base_name_arr = []
            # get effective dir name
            for i in os.path.split(file_name):
                if "--" in i:
                    base_name_arr.append(i)
                    break
                base_name_arr.append(i)
            base_name = os.path.join(*base_name_arr)
            safetensors_filename = base_name + "-merged"
        layer.base_layer._hf_hook.pre_forward(layer.base_layer)
        base_layer_offload = True

    yield

    for module in offloaded_modules:
        module._hf_hook.post_forward(module, torch.tensor([]))

    if base_layer_offload:
        # re-make weights map (must be on cpu to send params to the disk via memmap if disk offload)
        layer.base_layer._hf_hook.weights_map = {
            name: param.to("cpu") for name, param in named_module_tensors(layer.base_layer)
        }
        # offload weights map to disk if original device is the disk
        if torch.device("meta") in layer.base_layer._hf_hook.original_devices.values() and hasattr(
            layer.base_layer._hf_hook.weights_map, "dataset"
        ):
            # rewrite directory with merged weights
            offload_state_dict(safetensors_filename, layer.base_layer._hf_hook.weights_map)
        layer.base_layer._hf_hook.post_forward(layer.base_layer, torch.tensor([]))


def _check_lora_target_modules_mamba(peft_config: PeftConfig, model: nn.Module, target_name: str):
    """
    Prevent applying LoRA to incompatible modules in specific architectures (e.g., Mamba).
    """

    lora_like_types = {"LORA", "ADALORA", "XLORA", "RANDLORA"}
    incompatible_modules = {"out_proj", "conv1d"}
    mamba_model_types = {"falcon_h1", "mamba", "mamba2", "falcon_mamba"}

    if (
        peft_config.peft_type in lora_like_types
        and hasattr(model, "config")
        and getattr(model.config, "model_type", None) in mamba_model_types
    ):
        if target_name in incompatible_modules:
            raise ValueError(
                f"[PEFT:{peft_config.peft_type}] Module '{target_name}' is incompatible with Mamba-based models "
                f"(model_type='{model.config.model_type}'). Incompatible modules: {incompatible_modules}. "
                "Please remove it from `target_modules` to avoid compatibility issues."
            )


class BaseTuner(nn.Module, ABC):
    r"""
    A base tuner model that provides the common methods and attributes for all tuners that are injectable into a
    torch.nn.Module

    For adding a new Tuner class, one needs to overwrite the following methods:

    - **_prepare_adapter_config**:
        A private method to eventually prepare the adapter config, for example in case the field `target_modules` is
        missing.
    - **_create_and_replace**:
        A private method to create and replace the target module with the adapter module.
    - **_check_target_module_exists**:
        A private helper method to check if the passed module's key name matches any of the target modules in the
        adapter_config.

    The easiest is to check what is done in the `peft.tuners.lora.LoraModel` class.

    Attributes:
        model (`torch.nn.Module`):
            The model to which the adapter tuner layers will be attached.
        forward (`Callable`):
            The forward method of the model.
        peft_config (`Union[`PeftConfig`, dict[str, PeftConfig]]`):
            The adapter configuration object, it should be a dictionary of `str` to `PeftConfig` objects. One can also
            pass a PeftConfig object and a new adapter will be created with the default name `adapter` or create a new
            dictionary with a key `adapter_name` and a value of that peft config.
        config (`dict[str, Any]`):
            The model configuration object, it should be a dictionary of `str` to `Any` objects.
        targeted_module_names (`list[str]`):
            The list of module names that were actually adapted. Can be useful to inspect if you want to quickly
            double-check that the `config.target_modules` were specified correctly.
        targeted_parameter_names (`list[str]`):
            The list of parameter names that were actually adapted. Can be useful to inspect if you want to quickly
            double-check that the `config.target_parameters` were specified correctly.
    """

    def __init__(
        self,
        model,
        peft_config: Union[PeftConfig, dict[str, PeftConfig]],
        adapter_name: str,
        low_cpu_mem_usage: bool = False,
        state_dict: Optional[dict[str, torch.Tensor]] = None,
    ) -> None:
        super().__init__()

        self.model = model
        self.targeted_module_names: list[str] = []
        self.targeted_parameter_names: list[str] = []

        # For advanced developers, if you want to attach multiple adapters to your
        # model, just add a `peft_config` dict attribute to your model.
        if not hasattr(self, "peft_config"):
            self.peft_config = {adapter_name: peft_config} if isinstance(peft_config, PeftConfig) else peft_config
        else:
            warnings.warn(
                "Already found a `peft_config` attribute in the model. This will lead to having multiple adapters"
                " in the model. Make sure to know what you are doing!"
            )
            if isinstance(peft_config, PeftConfig):
                self.peft_config[adapter_name] = peft_config
            else:
                # user is adding a dict of PeftConfigs
                self.peft_config.update(peft_config)

        self.active_adapter: str | list[str] = adapter_name
        self._pre_injection_hook(self.model, self.peft_config[adapter_name], adapter_name)
        if peft_config != PeftType.XLORA or peft_config[adapter_name] != PeftType.XLORA:
            self.inject_adapter(self.model, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage, state_dict=state_dict)

        # Copy the peft_config in the injected model.
        self.model.peft_config = self.peft_config

    @property
    def active_adapters(self) -> list[str]:
        if isinstance(self.active_adapter, str):
            return [self.active_adapter]
        # is already a list of str
        return self.active_adapter

    def forward(self, *args: Any, **kwargs: Any):
        return self.model.forward(*args, **kwargs)

    def _pre_injection_hook(self, model: nn.Module, config: PeftConfig, adapter_name: str) -> None:
        r"""
        A hook to be called before the adapter is injected into the model. This method can be overridden by child
        classes to perform any pre-injection operations.

        Args:
            model (`nn.Module`):
                The model to be adapted.
            config (`PeftConfig`):
                The adapter config.
            adapter_name (`str`):
                The adapter name.
        """
        pass

    @abstractmethod
    def _prepare_adapter_config(self, peft_config: PeftConfig, model_config: dict) -> PeftConfig:
        r"""
        A private method to eventually prepare the adapter config. For transformers based models, if
        `peft_config.target_modules` is None, we can automatically infer the target modules from the
        `TRANSFORMERS_MODELS_TO_XXX_TARGET_MODULES_MAPPING`. This method can be further refactored in the future to
        automatically infer it for all tuner models.

        Check out `peft.tuner.lora.LoraModel._prepare_adapter_config` for an example.

        Args:
            peft_config (`PeftConfig`):
                The adapter config.
            model_config (`dict`):
                The transformers model config, that config should contain the `model_type` key.
        """
        ...

    def _prepare_model(self, peft_config: PeftConfig, model: nn.Module):
        r"""
        A private method to modify the model structure before adapter is applied.

        See `peft.tuner.lora.LoraModel._prepare_model` for an example.

        Args:
            peft_config (`PeftConfig`):
                The prepared adapter config.
            model (`nn.Module`):
                The model that is going to be adapted.
        """
        pass

    @abstractmethod
    def _check_target_module_exists(peft_config: PeftConfig, key: str) -> bool:
        r"""
        A helper private method to check if the passed module's key name matches any of the target modules in the
        `peft_config.target_modules` list. If it does, return `True`, else return `False`.

        Args:
            peft_config (`PeftConfig`):
                The adapter config.
            key (`str`):
                The module's key name.
        """
        ...

    @abstractmethod
    def _create_and_replace(
        self,
        peft_config: PeftConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
        parameter_name: Optional[str] = None,
    ) -> None:
        r"""
        Inplace replacement of the target module with the adapter layer. This method needs to be overridden by all the
        tuner classes.

        Check `peft.tuners.lora.LoraModel._create_and_replace` for an example.

        Args:
            peft_config (`PeftConfig`):
                The adapter config.
            adapter_name (`str`):
                The adapter name.
            target (`nn.Module`):
                The target module.
            target_name (`str`):
                The target module's name.
            parent (`nn.Module`):
                The parent module.
            current_key (`str`):
                The key of the current target being adapted.
            parameter_name (`str`, *optional*)
                If, and only if, an `nn.Parameter` is being targeted, this is the name of the parameter.
        """
        ...

    @abstractmethod
    def _mark_only_adapters_as_trainable(self, model: nn.Module):
        r"""
        A helper method to mark only the adapter layers as trainable (i.e. module.requires_grad = False) This needs to
        be overridden for all tuner classes to match the correct key names.

        Check `peft.tuners.lora.LoraModel._mark_only_adapters_as_trainable` for an example.
        """
        ...

    @abstractmethod
    def disable_adapter_layers(self) -> None:
        """
        Disable all adapters in-place.
        """
        ...

    @abstractmethod
    def enable_adapter_layers(self) -> None:
        """
        Enable all adapters in-place
        """
        ...

    def _check_new_adapter_config(self, config: PeftConfig) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.

        """
        pass

    def _cast_adapter_dtype(self, adapter_name: str, autocast_adapter_dtype: bool = True) -> None:
        """
        A helper method to cast the adapter weights to the correct dtype.

        Currently, this only upcasts float16 and bfloat16 to float32.

        Args:
            adapter_name (`str`):
                The adapter name.
            autocast_adapter_dtype (`bool`, *optional*):
                Whether to autocast the adapter dtype. Defaults to `True`.

        """
        if not autocast_adapter_dtype:
            return

        dtypes_to_convert_to_fp32 = {torch.float16, torch.bfloat16}

        for module in self.model.modules():
            if not isinstance(module, BaseTunerLayer):
                continue

            for submodule in module.modules():
                if not isinstance(submodule, (nn.ModuleDict, nn.ParameterDict, BufferDict)):
                    continue

                if adapter_name not in submodule:
                    continue

                if isinstance(submodule[adapter_name], nn.Parameter):
                    if submodule[adapter_name].dtype in dtypes_to_convert_to_fp32:
                        submodule[adapter_name].data = submodule[adapter_name].data.to(torch.float32)
                    continue

                if isinstance(submodule[adapter_name], torch.Tensor):  # e.g. from a BufferDict
                    if submodule[adapter_name].dtype in dtypes_to_convert_to_fp32:
                        submodule[adapter_name] = submodule[adapter_name].to(torch.float32)
                    continue

                for param in submodule[adapter_name].parameters():
                    if param.dtype in dtypes_to_convert_to_fp32:
                        param.data = param.data.to(torch.float32)

    def _check_merge_allowed(self):
        """Helper method to check whether the adapter can be merged.

        Raise a ValueError if it is not possible to merge the adapter with the given configuration.
        """
        example_code = textwrap.dedent(
            """
            ```python
            from transformers import AutoModelForCausalLM

            # Load original tied model
            model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it", tie_word_embeddings=False)

            # Set the randomly initialized lm_head to the previously tied embeddings
            model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

            # Save the untied model
            untied_model_dir = "dir/for/untied/model"
            model.save_pretrained(untied_model_dir)
            model.config.save_pretrained(untied_model_dir)

            # Now use the original model but in untied format
            model = AutoModelForCausalLM.from_pretrained(untied_model_dir)
            ```
            """
        )
        tied_target_modules = self._get_tied_target_modules(self.model)
        if tied_target_modules:
            warnings.warn(
                f"Model with `tie_word_embeddings=True` and the {tied_target_modules=} are part of the adapter. "
                "This can lead to complications. "
                "You can opt to merge the adapter after cloning the weights (to untie the embeddings). "
                "You can untie the embeddings by loading the model with `tie_word_embeddings=False`. For example:"
                + example_code
            )

    def _check_target_module_compatiblity(self, peft_config: PeftConfig, model: nn.Module, target_name: str):
        """
        Prevent applying LoRA to incompatible modules in specific architectures (e.g., Mamba).
        """
        _check_lora_target_modules_mamba(peft_config, model, target_name)

    def _create_and_replace_parameter(
        self, peft_config, adapter_name, target, target_name, parent, current_key
    ) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} does not support targeting nn.Parameter.")

    def inject_adapter(
        self,
        model: nn.Module,
        adapter_name: str,
        autocast_adapter_dtype: bool = True,
        low_cpu_mem_usage: bool = False,
        state_dict: Optional[dict[str, torch.Tensor]] = None,
    ) -> None:
        r"""
        Creates adapter layers and replaces the target modules with the adapter layers. This method is called under the
        hood by `peft.mapping.get_peft_model` if a non-prompt tuning adapter class is passed.

        The corresponding PEFT config is directly retrieved from the `peft_config` attribute of the BaseTuner class.

        Args:
            model (`nn.Module`):
                The model to be tuned.
            adapter_name (`str`):
                The adapter name.
            autocast_adapter_dtype (`bool`, *optional*):
                Whether to autocast the adapter dtype. Defaults to `True`.
            low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
                Create empty adapter weights on meta device. Useful to speed up the loading process.
            state_dict (`dict`, *optional*, defaults to `None`)
                If a state_dict is passed here, the adapters will be injected based on the entries of the state_dict.
                This can be useful when the exact `target_modules` of the PEFT method is unknown, for instance because
                the checkpoint was created without meta data. Note that the values from the state_dict are not used,
                only the keys are used to determine the correct layers that should be adapted.

        """
        ###################################
        # PREPARATION OF MODEL AND CONFIG #
        ###################################

        peft_config = self.peft_config[adapter_name]
        excluded_modules = []
        unmatched_modules = []
        targeted_modules_from_peft_config: list[str] = []  # only relevant if state_dict is passed
        # Note: If possible, all checks should be performed *at the start of this method*.
        # This way, we can raise early if something goes wrong, without leaving the model
        # in a bad (half-initialized) state.
        self._check_new_adapter_config(peft_config)

        model_config = self.get_model_config(model)

        peft_config = self._prepare_adapter_config(peft_config, model_config)

        self._prepare_model(peft_config, model)

        if getattr(peft_config, "target_parameters", []) and state_dict:
            raise ValueError(
                "Trying to inject a PEFT adapter from a state_dict but the PEFT config uses `target_parameters`. This "
                "is not supported -- when using `target_parameters`, please inject the adapter without the state_dict."
            )

        named_modules = list(model.named_modules())
        key_list = [key for key, _ in named_modules]

        uses_dummy_target_modules = getattr(peft_config, "target_modules", None) == DUMMY_TARGET_MODULES
        if uses_dummy_target_modules:
            # dummy adapter, we allow not matching any module
            named_modules = []
            key_list = []

        # update peft_config.target_modules if required
        peft_config = _maybe_include_all_linear_layers(peft_config, model)

        # This is an optimization to reduce the number of entries in the target_modules list. The reason is that in some
        # circumstances, target_modules can contain hundreds of entries. Since each target module is checked against
        # each module of the net (which can be thousands), this can become quite expensive when many adapters are being
        # added. Often, the target_modules can be condensed in such a case, which speeds up the process.
        # A context in which this can happen is when diffusers loads non-PEFT LoRAs. As there is no meta info on
        # target_modules in that case, they are just inferred by listing all keys from the state_dict, which can be
        # quite a lot. See: https://github.com/huggingface/diffusers/issues/9297
        # As there is a small chance for undiscovered bugs, we apply this optimization only if the list of
        # target_modules is sufficiently big.
        # We also exclude IA³ from this optimization. This is because IA³ has both target_modules and
        # feedforward_modules, which are coupled (the latter must be a subset). It would be possible to change the logic
        # to keep both in sync, but it's not quite trivial and probably not worth the effort. See #2429.
        if (
            isinstance(peft_config.target_modules, (list, set))
            and (len(peft_config.target_modules) >= MIN_TARGET_MODULES_FOR_OPTIMIZATION)
            and (peft_config.peft_type != PeftType.IA3)
        ):
            names_no_target = [
                name
                for name in key_list
                if not any((name == suffix) or name.endswith("." + suffix) for suffix in peft_config.target_modules)
            ]
            new_target_modules = _find_minimal_target_modules(peft_config.target_modules, names_no_target)
            if len(new_target_modules) < len(peft_config.target_modules):
                peft_config.target_modules = new_target_modules

        ###############################
        # MATCHING & CREATING MODULES #
        ###############################

        existing_adapter_map = {}
        for key, module in named_modules:
            if isinstance(module, BaseTunerLayer):
                existing_adapter_map[key] = module

        # TODO: check if this the most robust way
        module_names: set[str] = set()
        if state_dict is not None:
            prefix = PEFT_TYPE_TO_PREFIX_MAPPING[peft_config.peft_type]
            module_names = {k.rsplit("." + prefix, 1)[0] for k in state_dict}

        for key, module in named_modules:
            if not key:
                continue

            # It is possible that we're adding an additional adapter, so if we encounter a key that clearly belongs to a
            # previous adapter we can skip here since we don't want to interfere with adapter internals.
            for adapter_key in existing_adapter_map:
                if key.startswith(adapter_key + "."):
                    excluded_modules.append(key)
                    break

            if excluded_modules and excluded_modules[-1] == key:
                continue

            if state_dict is None:
                # normal mechanism: match the modules using the peft_config
                result = self._check_target_module_exists(peft_config, key)
                if isinstance(result, _ExcludedModule):
                    excluded_modules.append(key)
                elif not result:
                    unmatched_modules.append(key)
                else:
                    self.targeted_module_names.append(key)
                    parent, target, target_name = _get_submodules(model, key)
                    self._check_target_module_compatiblity(peft_config, model, target_name)
                    ctx = init_empty_weights if low_cpu_mem_usage else nullcontext
                    with ctx():
                        self._create_and_replace(
                            peft_config, adapter_name, target, target_name, parent, current_key=key
                        )
            else:
                # use the state_dict to match modules instead
                if key not in module_names:
                    unmatched_modules.append(key)
                else:
                    self.targeted_module_names.append(key)
                    parent, target, target_name = _get_submodules(model, key)
                    self._check_target_module_compatiblity(peft_config, model, target_name)
                    ctx = init_empty_weights if low_cpu_mem_usage else nullcontext
                    with ctx():
                        self._create_and_replace(
                            peft_config, adapter_name, target, target_name, parent, current_key=key
                        )

                # still record what would have been matched via the config so that the two results can be compared
                if self._check_target_module_exists(peft_config, key):
                    targeted_modules_from_peft_config.append(key)

        if getattr(peft_config, "target_parameters", []):
            # Note: We don't need to check for no state_dict being passed, since we already checked this earlier.
            self._inject_parameters(
                peft_config=peft_config, model=model, adapter_name=adapter_name, low_cpu_mem_usage=low_cpu_mem_usage
            )

        ####################
        # CHECK FOR ERRORS #
        ####################

        if state_dict is not None:
            # in case that the state_dict was used as source of truth and it resulted in different outcomes than what
            # would have been matched with the PEFT config, warn the user about that.
            targeted_set_from_peft_config = set(targeted_modules_from_peft_config)
            targeted_set_from_state_dict = set(self.targeted_module_names)
            diff_peft_config = targeted_set_from_peft_config - targeted_set_from_state_dict
            diff_state_dict = targeted_set_from_state_dict - targeted_set_from_peft_config
            warning_msg = ""
            if diff_peft_config or diff_state_dict:
                warning_msg = (
                    "While injecting the PEFT adapters, an inconsistency was discovered between the PEFT config and "
                    "the provided state_dict. This is not necessarily an issue and can be ignored if this was the "
                    "intent. "
                )
            if diff_peft_config:
                warning_msg += (
                    f"The PEFT config contained these additional target modules: {sorted(diff_peft_config)}. "
                )
            if diff_state_dict:
                warning_msg += f"The state_dict contained these additional target modules: {sorted(diff_state_dict)}. "
            if warning_msg:
                warnings.warn(warning_msg, RuntimeWarning)

        if not self.targeted_module_names and not self.targeted_parameter_names and not uses_dummy_target_modules:
            if excluded_modules and not unmatched_modules:
                # All targeted modules were excluded
                raise ValueError(
                    "All modules were excluded. This is likely unintended. "
                    "Check your `target_modules`, `exclude_modules` and `modules_to_save` configuration."
                )
            elif not excluded_modules and unmatched_modules and not peft_config.target_modules:
                raise ValueError(
                    "No `target_modules` passed but also no `target_parameters` found. Please check the values for "
                    "these arguments."
                )
            elif not excluded_modules and unmatched_modules:
                # None of the targeted modules matched
                error_msg = (
                    f"Target modules {peft_config.target_modules} not found in the base model. "
                    f"Please check the target modules and try again."
                )
                if getattr(peft_config, "layers_to_transform", None) is not None:
                    error_msg += f" Note: You specified 'layers_to_transform': {peft_config.layers_to_transform}."
                if getattr(peft_config, "layers_pattern", None) is not None:
                    error_msg += f" You also specified 'layers_pattern': {peft_config.layers_pattern}."
                raise ValueError(error_msg)
            else:
                # Some modules did not match and some matched but were excluded
                error_msg = (
                    "No modules were targeted for adaptation. "
                    "This might be caused by a combination of mismatched target modules and excluded modules. "
                    "Please check your `target_modules` and `exclude_modules` configuration. You may also have "
                    "only targeted modules that are marked to be saved (`modules_to_save`)."
                )
                if getattr(peft_config, "layers_to_transform", None) is not None:
                    error_msg += f" Note: You specified 'layers_to_transform': {peft_config.layers_to_transform}."
                if getattr(peft_config, "layers_pattern", None) is not None:
                    error_msg += f" You also specified 'layers_pattern': {peft_config.layers_pattern}."
                raise ValueError(error_msg)

        elif hasattr(peft_config, "exclude_modules") and peft_config.exclude_modules and not excluded_modules:
            # exclude_modules was passed but was not used
            warnings.warn(
                f"You have passed exclude_modules={peft_config.exclude_modules} but no modules were excluded. "
                "Please check that exclude_modules was set correctly."
            )

        elif not uses_dummy_target_modules:
            # If we landed here, it means that at least one module or parameter was adapted, so let's not raise an
            # error. However, let's warn the user if it seems like
            # - they wanted to match a module but there was no match
            # - they wanted to match a parameter but there was no match
            if peft_config.target_modules and not self.targeted_module_names:
                warnings.warn(
                    f"target_modules={peft_config.target_modules} were set but no module was matched.", RuntimeWarning
                )
            elif getattr(peft_config, "target_parameters", []) and not self.targeted_parameter_names:
                warnings.warn(
                    f"target_parameters={peft_config.target_parameters} were set but no parameter was matched.",
                    RuntimeWarning,
                )

        tied_target_modules = self._get_tied_target_modules(model=model)
        if tied_target_modules:
            warnings.warn(
                f"Model with `tie_word_embeddings=True` and the {tied_target_modules=} are part of the adapter. "
                "This can lead to complications, for example when merging the adapter "
                "or converting your model to formats other than safetensors. "
                "See for example https://github.com/huggingface/peft/issues/2018."
            )

        ################
        # HOUSEKEEPING #
        ################

        # It's important to set the adapter here (again), because otherwise it can happen that if a 2nd adapter is
        # added, and it targets different layer(s) than the first adapter (which is active), then those different
        # layers will be activated, which we don't want.
        self.set_adapter(self.active_adapters)
        self._mark_only_adapters_as_trainable(model)

        if self.peft_config[adapter_name].inference_mode:
            for n, p in model.named_parameters():
                if adapter_name in n:
                    p.requires_grad = False

        set_additional_trainable_modules(
            model=model,
            peft_config=peft_config,
            model_config=BaseTuner.get_model_config(self),
            adapter_name=adapter_name,
        )

    def _inject_parameters(
        self, peft_config: PeftConfig, model: nn.Module, adapter_name: str, low_cpu_mem_usage: bool
    ) -> None:
        # TODO very simple matching, might not cover all use cases
        target_names = set(peft_config.target_parameters)
        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                # It is possible that the layer is already a PEFT layer and needs updating with a new adapter. In this
                # case, the name of parameter would be something like `model.layers.0.experts.base_layer.weight`, i.e.
                # there is a "base_layer" inserted in the name. We need to remove that, otherwise we won't be able to
                # match correctly (in this case, "experts.weight" would not match).
                prefix, _, suffix = module_name.rpartition(".base_layer")
                module_name = prefix + suffix
                key = f"{module_name}.{param_name}"
                # we're interested in finding the "lowest" module that contains the parameter, hence recurse=False
                if (key in target_names) or any(key.endswith(f".{target_key}") for target_key in target_names):
                    self.targeted_parameter_names.append(key)

                    parent, target, target_name = _get_submodules(model, module_name)
                    # use the class name for checking to avoid circular import
                    if isinstance(target, BaseTunerLayer) and target.__class__.__name__ != "ParamWrapper":
                        raise ValueError(
                            f"Trying to wrap an `nn.Parameter` of layer '{target_name}' of type "
                            f"{type(target).__name__}, which is not a valid target. Make sure that this layer is not "
                            "also targeted with `target_modules`. For some models, PEFT will do this automatically, "
                            "try setting `target_modules=[]` to prevent it."
                        )

                    self._check_target_module_compatiblity(peft_config, model, target_name)
                    ctx = init_empty_weights if low_cpu_mem_usage else nullcontext
                    with ctx():
                        self._create_and_replace(
                            peft_config,
                            adapter_name,
                            target,
                            target_name,
                            parent,
                            current_key=key,
                            parameter_name=param_name.rpartition(".")[-1],
                        )

    def merge_adapter(self, adapter_names: Optional[list[str]] = None, safe_merge: bool = False) -> None:
        """
        This method merges the adapter layers into the base model.

        Merging adapters can lead to a speed up of the forward pass. A copy of the adapter weights is still kept in
        memory, which is required to unmerge the adapters. In order to merge the adapter weights without keeping them
        in memory, please call `merge_and_unload`.

        Args:
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If `None`, all active adapters will be merged.
                Defaults to `None`.
            safe_merge (`bool`, *optional*):
                If `True`, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
        """
        # Note: The order of arguments here is:
        #   adapter_names, safe_merge
        # For layer.merge, the order is:
        #   safe_merge, adapter_names
        # This is not so nice but this method here started with only adapter_names, thus putting safe_merge first would
        # be a backwards incompatible change.
        self._check_merge_allowed()
        for module in self.model.modules():
            if isinstance(module, BaseTunerLayer):
                with onload_layer(module):
                    module.merge(adapter_names=adapter_names, safe_merge=safe_merge)

    def unmerge_adapter(self):
        """
        This method unmerges all merged adapter layers from the base model.
        """
        for module in self.model.modules():
            if isinstance(module, BaseTunerLayer):
                with onload_layer(module):
                    module.unmerge()

    def _delete_auxiliary_adapter(self, adapter_name: str, new_active_adapters: Optional[list[str]]) -> None:
        for module in self.modules():
            if isinstance(module, AuxiliaryTrainingWrapper):
                module.delete_adapter(adapter_name, new_active_adapters=new_active_adapters)

    def _unloading_checks(self, adapter_names: Optional[list[str]]):
        adapters_to_consider = adapter_names or self.active_adapters
        is_modules_to_save_available = any(
            self.peft_config[adapter].modules_to_save for adapter in adapters_to_consider
        )
        if is_modules_to_save_available and len(adapters_to_consider) > 1:
            raise ValueError("Cannot unload multiple adapters that specify `modules_to_save`.")

    @staticmethod
    def get_model_config(model: nn.Module) -> dict:
        """
        This method gets the config from a model in dictionary form. If model has not attribute config, then this
        method returns a default config.

        Args:
            model (`nn.Module`):
                Model to get the config from.
            default (`dict|None`, *optional*)::
                What to return if model does not have a config attribute.
        """
        model_config = getattr(model, "config", DUMMY_MODEL_CONFIG)
        if hasattr(model_config, "to_dict"):
            model_config = model_config.to_dict()
        return model_config

    def _get_tied_target_modules(self, model: nn.Module) -> list[str]:
        tied_target_modules = []
        model_config = self.get_model_config(model)
        if model_config.get("tie_word_embeddings"):
            for target_module in self.targeted_module_names:
                # This potentially yields false positives since we're just looking at the layer names. So if we use a
                # model that uses weight-tying of lm_head and embed_tokens, a third, unrelated, layer which is
                # unfortunately named so that it is in EMBEDDING_LAYER_NAMES will be falsely reported here as well.
                if target_module.split(".")[-1] in EMBEDDING_LAYER_NAMES:
                    tied_target_modules.append(target_module)
        return tied_target_modules


class BaseTunerLayer(ABC):
    r"""
    A tuner layer mixin that provides the common methods and attributes for all tuners.

    Args:
        is_pluggable (`bool`, *optional*):
            Whether the adapter layer can be plugged to any pytorch module
        active_adapters (Union[List[`str`], `str`], *optional*):
            The name of the active adapter.
    """

    # All names of layers that may contain adapter (trainable) weights
    adapter_layer_names: tuple[str, ...] = ()
    # All names of other parameters that may contain adapter-related parameters
    other_param_names: tuple[str, ...] = ()

    # indicates whether all adapters should be disabled
    _disable_adapters: bool = False

    # the currently active adapter(s)
    _active_adapter: str | list[str] = "default"

    # List all merged adapters
    merged_adapters: list[str] = []

    def get_base_layer(self) -> nn.Module:
        """
        (Recursively) get the base_layer.

        This is necessary for the case that the tuner layer wraps another tuner layer.

        """
        base_layer = self
        while hasattr(base_layer, "base_layer"):
            base_layer = base_layer.base_layer
        return base_layer

    @property
    def weight(self) -> torch.Tensor:
        # This is required for some transformers code, e.g. for T5, weight is accessed as:
        #     self.wo.weight
        # where "wo" is the adapter layer.
        # https://github.com/huggingface/transformers/blob/78f6ed6c70b29c1560780e3869a7ad4c6b3d2710/src/transformers
        # /models/t5/modeling_t5.py#L292
        base_layer = self.get_base_layer()
        if hasattr(base_layer, "qweight"):
            # QuantLinear
            weight = base_layer.qweight
        else:
            # Other layers
            weight = base_layer.weight
        return weight

    @property
    def bias(self) -> torch.Tensor:
        base_layer = self.get_base_layer()
        return base_layer.bias

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        raise NotImplementedError

    def unmerge(self) -> None:
        raise NotImplementedError

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    @property
    def disable_adapters(self) -> bool:
        # use a property to ensure that disable_adapters is not set directly, instead use the enable_adapters method
        return self._disable_adapters

    @property
    def active_adapter(self) -> str | list[str]:
        # use a property to ensure that active_adapter is not set directly, instead use the set_adapter method
        return self._active_adapter

    def _get_available_adapters(self) -> set[str]:
        """Return all adapter names that can be found on this module."""
        adapters = set()
        for layer_name in self.adapter_layer_names:
            module = getattr(self, layer_name)
            if not isinstance(module, (nn.ModuleDict, nn.ParameterDict)):
                continue
            adapters.update(set(module.keys()))
        return adapters

    @property
    def active_adapters(self):
        if isinstance(self.active_adapter, str):
            return [self.active_adapter]
        # is already a list of str
        return self.active_adapter

    def enable_adapters(self, enabled: bool) -> None:
        """Toggle the enabling and disabling of adapters

        Takes care of setting the requires_grad flag for the adapter weights.

        Args:
            enabled (bool): True to enable adapters, False to disable adapters
        """
        if enabled:
            self.set_adapter(self.active_adapters)
            self._disable_adapters = False
        else:
            # disable grads on all adapter layers
            for layer_name in self.adapter_layer_names:
                layer = getattr(self, layer_name)
                layer.requires_grad_(False)
            self._disable_adapters = True

    def set_adapter(self, adapter_names: str | list[str]) -> None:
        """Set the active adapter(s).

        Additionally, this function will set the specified adapters to trainable (i.e., requires_grad=True). If this is
        not desired, use the following code.

        ```py
        >>> for name, param in model_peft.named_parameters():
        ...     if ...:  # some check on name (ex. if 'lora' in name)
        ...         param.requires_grad = False
        ```

        Args:
            adapter_name (`str` or `List[str]`): Name of the adapter(s) to be activated.
        """
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        # Deactivate grads on the inactive adapter and activate grads on the active adapter
        for layer_name in self.adapter_layer_names:
            module_dict = getattr(self, layer_name)
            for key, layer in module_dict.items():
                if key in adapter_names:
                    # Note: It is possible that not a single layer is called with requires_grad_(True) here. This may
                    # happen if a completely different adapter layer is being activated.
                    layer.requires_grad_(True)
                else:
                    layer.requires_grad_(False)

        self._active_adapter = adapter_names

    def _all_available_adapter_names(self) -> list[str]:
        """Return a sorted list of all available adapter names"""
        adapter_names = set()
        for name in self.adapter_layer_names + self.other_param_names:
            # we check each possible attribute and if it's a dict or ModuleDict, we assume that the keys are the adapter
            # names
            attr = getattr(self, name)
            if hasattr(attr, "keys"):
                adapter_names.update(attr.keys())
        return sorted(adapter_names)

    def delete_adapter(self, adapter_name: str) -> None:
        """
        Delete an adapter from the layer

        This should be called on all adapter layers, or else we will get an inconsistent state.

        This method will also set a new active adapter if the deleted adapter was an active adapter. It is important
        that the new adapter is chosen in a deterministic way, so that the same adapter is chosen on all layers.

        Args:
            adapter_name (`str`): The name of the adapter to delete

        """
        for attr in self.adapter_layer_names + self.other_param_names:
            if adapter_name in getattr(self, attr):
                del getattr(self, attr)[adapter_name]

        if adapter_name in self.active_adapters:
            # choose a new active adapter
            active_adapters = self.active_adapters[:]
            active_adapters.remove(adapter_name)
            if active_adapters:
                self.set_adapter(active_adapters)
            else:
                # no active adapters left, set a new default adapter
                # here we get the list of all adapters existing adapter names and choose the first one
                remaining_adapters = self._all_available_adapter_names()
                if not remaining_adapters:
                    self.set_adapter([])
                else:
                    new_active_adapter = remaining_adapters[0]
                    warnings.warn(
                        f"Adapter {adapter_name} was active which is now deleted. Setting active adapter to "
                        f"{new_active_adapter}."
                    )
                    self.set_adapter(remaining_adapters[0])

    def _move_adapter_to_device_of_base_layer(self, adapter_name: str, device: Optional[torch.device] = None) -> None:
        """
        Move the adapter of the given name to the device of the base layer.
        """
        if device is None:
            base_layer = self.get_base_layer()
            if isinstance(base_layer, nn.MultiheadAttention):
                base_layer = base_layer.out_proj
            # check weight and qweight (for GPTQ)
            for weight_name in ("weight", "qweight"):
                weight = getattr(base_layer, weight_name, None)
                if weight is not None:
                    device = weight.device
                    dtype = weight.dtype
                    break
            else:
                # no break encountered: could not determine the device
                return

        meta = torch.device("meta")

        # loop through all potential adapter layers and move them to the device of the base layer; be careful to only
        # move this specific adapter to the device, as the other adapters could be on different devices
        # see #1639
        for adapter_layer_name in self.adapter_layer_names + self.other_param_names:
            adapter_layer = getattr(self, adapter_layer_name, None)
            if not isinstance(adapter_layer, (nn.ModuleDict, nn.ParameterDict, BufferDict)):
                continue
            if adapter_name not in adapter_layer:
                continue
            if any(p.device == meta for p in adapter_layer.parameters()):
                continue

            # TODO: weight is not necessarily defined here, leading to a NameError, fix that
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                adapter_layer[adapter_name] = adapter_layer[adapter_name].to(device, dtype=dtype)
            else:
                adapter_layer[adapter_name] = adapter_layer[adapter_name].to(device)

    @overload
    def _cast_input_dtype(self, x: None, dtype: torch.dtype) -> None: ...

    @overload
    def _cast_input_dtype(self, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor: ...

    def _cast_input_dtype(self, x, dtype: torch.dtype):
        """
        Whether to cast the dtype of the input of the forward method.

        Usually, we want to enable this to align the input dtype with the dtype of the weight, but by setting
        layer.cast_input_dtype=False, this can be disabled if necessary.

        Enabling or disabling can be managed via the peft.helpers.disable_lora_input_dtype_casting context manager.
        """
        if x is None:  # useful e.g. if x is the bias, which can be None
            return None

        cast_input_dtype_enabled = getattr(self, "cast_input_dtype_enabled", True)
        if (not cast_input_dtype_enabled) or (x.dtype == dtype):
            return x
        return x.to(dtype=dtype)


def _find_minimal_target_modules(
    target_modules: list[str] | set[str], other_module_names: list[str] | set[str]
) -> set[str]:
    """Find the minimal set of target modules that is sufficient to separate them from the other modules.

    Sometimes, a very large list of target_modules could be passed, which can slow down loading of adapters (e.g. when
    loaded from diffusers). It may be possible to condense this list from hundreds of items to just a handful of
    suffixes that are sufficient to distinguish the target modules from the other modules.

    Example:
        ```py
        >>> from peft.tuners.tuners_utils import _find_minimal_target_modules

        >>> target_modules = [f"model.decoder.layers.{i}.self_attn.q_proj" for i in range(100)]
        >>> target_modules += [f"model.decoder.layers.{i}.self_attn.v_proj" for i in range(100)]
        >>> other_module_names = [f"model.encoder.layers.{i}.self_attn.k_proj" for i in range(100)]
        >>> _find_minimal_target_modules(target_modules, other_module_names)
        {"q_proj", "v_proj"}
        ```

    Args:
        target_modules (`list[str]` | `set[str]`):
            The list of target modules.
        other_module_names (`list[str]` | `set[str]`):
            The list of other module names. They must not overlap with the target modules.

    Returns:
        `set[str]`:
            The minimal set of target modules that is sufficient to separate them from the other modules.

    Raises:
        ValueError:
            If `target_modules` is not a list or set of strings or if it contains an empty string. Also raises an error
            if `target_modules` and `other_module_names` contain common elements.
    """
    if isinstance(target_modules, str) or not target_modules:
        raise ValueError("target_modules should be a list or set of strings.")

    target_modules = set(target_modules)
    if "" in target_modules:
        raise ValueError("target_modules should not contain an empty string.")

    other_module_names = set(other_module_names)
    if not target_modules.isdisjoint(other_module_names):
        msg = (
            "target_modules and other_module_names contain common elements, this should not happen, please "
            "open a GitHub issue at https://github.com/huggingface/peft/issues with the code to reproduce this issue"
        )
        raise ValueError(msg)

    # it is assumed that module name parts are separated by a "."
    def generate_suffixes(s):
        parts = s.split(".")
        return [".".join(parts[i:]) for i in range(len(parts))][::-1]

    # Create a reverse lookup for other_module_names to quickly check suffix matches
    other_module_suffixes = {suffix for item in other_module_names for suffix in generate_suffixes(item)}

    # Find all potential suffixes from target_modules
    target_modules_suffix_map = {item: generate_suffixes(item) for item in target_modules}

    # Initialize a set for required suffixes
    required_suffixes = set()

    # We sort the target_modules_suffix_map simply to get deterministic behavior, since sets have no order. In theory
    # the order should not matter but in case there is a bug, it's better for the bug to be deterministic.
    for item, suffixes in sorted(target_modules_suffix_map.items(), key=lambda tup: tup[1]):
        # Go through target_modules items, shortest suffixes first
        for suffix in suffixes:
            # If the suffix is already in required_suffixes or matches other_module_names, skip it
            if suffix in required_suffixes or suffix in other_module_suffixes:
                continue
            # Check if adding this suffix covers the item
            if not any(item.endswith("." + req_suffix) for req_suffix in required_suffixes):
                required_suffixes.add(suffix)
                break

    if not required_suffixes:
        return set(target_modules)
    return required_suffixes


class _ExcludedModule:
    """
    A private helper method used to represent excluded modules in the check_target_module_exists function.
    """

    def __bool__(self):
        return False


def check_target_module_exists(config, key: str) -> bool | re.Match[str] | None:
    """A helper method to check if the passed module's key name matches any of the target modules in the adapter_config.

    Args:
        config (`LoraConfig` | `LycorisConfig`): A config to match target modules from
        key (`str`): A key to search any matches in config

    Returns:
        `bool` | `re.Match[str]` | `None`: True of match object if key matches any target modules from config, False or
        None if no match found
    """
    if hasattr(config, "exclude_modules") and config.exclude_modules:
        if isinstance(config.exclude_modules, str):
            if re.fullmatch(config.exclude_modules, key):
                return _ExcludedModule()
        elif key in config.exclude_modules:
            return _ExcludedModule()
        elif any(key.endswith(f".{exclude_key}") for exclude_key in config.exclude_modules):
            return _ExcludedModule()

    # Adapters should never match on modules to save modules as it is a guarantee for conflicts of behavior
    # between `ModulesToSaveWrapper` internals and the potential adapter.
    modules_to_save = getattr(config, "modules_to_save", None)
    if modules_to_save:
        if any(re.match(rf"(^|.*\.){m}($|\..*)", key) for m in modules_to_save):
            return _ExcludedModule()

    if (config.target_modules is None) and (config.target_parameters is not None):
        # this is allowed if config.target_parameters are specified
        return False

    if isinstance(config.target_modules, str):
        target_module_found = match_target_against_key(config.target_modules, key)
    elif key in config.target_modules:
        # this module is specified directly in target_modules
        target_module_found = True
    else:
        target_module_found = any(key.endswith(f".{target_key}") for target_key in config.target_modules)

        layer_indexes = getattr(config, "layers_to_transform", None)
        layers_pattern = getattr(config, "layers_pattern", None)

        is_using_layer_indexes = layer_indexes is not None and (
            len(layer_indexes) != 0 if isinstance(layer_indexes, list) else True
        )
        if is_using_layer_indexes and target_module_found:
            layer_index = None
            # TODO: It's still unclear how empty layers_pattern (None, [], or "") should behave
            # For now, empty layers_pattern means any layer pattern is ok
            if layers_pattern is None or len(layers_pattern) == 0:
                layer_index = re.match(r".*\.[^.]*\.(\d+)\.", key)
            else:
                layers_pattern = [layers_pattern] if isinstance(layers_pattern, str) else layers_pattern
                for pattern in layers_pattern:
                    layer_index = re.match(rf".*\.{pattern}\.(\d+)\.", key)
                    if layer_index is not None:
                        break

            if layer_index is None:
                target_module_found = False
            else:
                layer_index = int(layer_index.group(1))
                if isinstance(layer_indexes, int):
                    target_module_found = layer_index == layer_indexes
                else:
                    target_module_found = layer_index in layer_indexes

    return target_module_found


def inspect_matched_modules(tuner: BaseTuner, adapter_name: str = "default") -> dict:
    """
    A helper function to inspect the set of matched and unmatched modules for a PEFT model and the given adapter.
    """
    config = tuner.peft_config[adapter_name]
    key_list = [key for key, _ in tuner.model.named_modules()]
    module_dict = {"matched": [], "unmatched": []}
    for key in key_list:
        if tuner._check_target_module_exists(config, key):
            module_dict["matched"].append(key)
        else:
            module_dict["unmatched"].append(key)
    return module_dict


def _maybe_include_all_linear_layers(peft_config: PeftConfig, model: nn.Module) -> PeftConfig:
    """
    Helper function to update `target_modules` to all linear/Conv1D layers if provided as 'all-linear'. Adapted from
    the QLoRA repository: https://github.com/artidoro/qlora/blob/main/qlora.py
    """
    if not hasattr(peft_config, "target_modules"):
        return peft_config

    # if `target_modules` is a string, convert to lower case and check if it matches "all-linear"
    if not (
        isinstance(peft_config.target_modules, str)
        and peft_config.target_modules.lower() == INCLUDE_LINEAR_LAYERS_SHORTHAND
    ):
        return peft_config

    linear_classes = (torch.nn.Linear, Conv1D)
    linear_names = ("Linear",)
    linear_module_names = set()
    for name, module in model.named_modules():
        # match with all linear classes.
        if isinstance(module, linear_classes):
            linear_module_names.add(name)
        elif isinstance(module, BaseTunerLayer) and any(n in type(module).__name__ for n in linear_names):
            # If the model already has adapter layers applied, then the "linear" layer is actually an adapter layer,
            # e.g. lora.Linear, and not nn.Linear. To target this layer, we don't want to check the layer type, as there
            # are many possible layer types (one for each PEFT method) and the list would quickly get out of date. Thus
            # we rely on the name of the layer class, which by convention is something like "Linear", "Linear4bit",
            # "HqqLoraLinear", ... in PEFT. It's not pretty but should generally work.
            # See 2390
            linear_module_names.add(name)

    # Try to remove linear layers that should not be targeted as best as possible. We have to rely on convention as
    # there are no hard rules to detect these modules.
    module_names_to_exclude = set()
    if isinstance(model, PreTrainedModel):
        output_emb = model.get_output_embeddings()
        if output_emb is not None:
            # ignore the last classification head for text generation models
            last_module_name = [name for name, module in model.named_modules() if module is output_emb][0]
            module_names_to_exclude.add(last_module_name)
        elif peft_config.task_type == TaskType.SEQ_CLS:
            # ignore classifier head for classification models (issue 2027)
            # there is no fix name for the classifier head, so check the common ones
            for name in SEQ_CLS_HEAD_NAMES:
                cls_head = getattr(model, name, None)
                if cls_head is not None:
                    last_module_name = [name for name, module in model.named_modules() if module is cls_head][0]
                    module_names_to_exclude.add(last_module_name)
                    break

    # we don't want nested LoRA layers, i.e. LoRA being applied to possibly existing lora_A, lora_B, etc.
    # see 2390
    for prefix, module in model.named_modules():
        if isinstance(module, BaseTunerLayer):
            for suffix, child in module.named_modules():
                if suffix:
                    module_names_to_exclude.add(f"{prefix}.{suffix}")

    linear_module_names -= module_names_to_exclude
    peft_config.target_modules = linear_module_names
    return peft_config


def check_adapters_to_merge(module: BaseTunerLayer, adapter_names: Optional[list[str]] = None) -> list[str]:
    """
    Helper function to check which adapters should be merged.

    Only return those adapters that are not already merged. Give a warning if some or all of the adapters are already
    merged.

    """
    if adapter_names is None:
        adapter_names = module.active_adapters
    if isinstance(adapter_names, str):
        raise ValueError(f"adapter_names should be a list of strings, got {adapter_names!r}.")

    if module.merged:
        merged_adapters = set(module.merged_adapters)
        adapter_names = [name for name in adapter_names if name not in merged_adapters]

        if adapter_names:
            warnings.warn(
                f"Already following adapters were merged {','.join(module.merged_adapters)}. "
                f"You are now additionally merging {','.join(adapter_names)}."
            )
        else:
            warnings.warn("All adapters are already merged, nothing to do.")

    return adapter_names


def clone_module(module: nn.Module, share_weights=False):
    """Clone a module in a pytorch model.

    Clones a module of a model, optionally sharing all the parameters between the original and the clone. Simplifies
    reusing a module when manipulating the architecture of a model.
    """
    clone = copy.deepcopy(module)

    def _share_weights(src: nn.Module, dst: nn.Module):
        for name, param in src.named_parameters(recurse=False):
            dst.register_parameter(name, param)

    if share_weights:
        for name, submodule in module.named_modules():
            _share_weights(submodule, clone.get_submodule(name))

    return clone


def replicate_layers(model: nn.Module, layer_map: list[tuple[int, int]]):
    """Replicate layers in a transfomer model with weight sharing.

    This function looks for a module list attribute at model[(.model)*].layers and replicates the layers in the module
    list according to the layer map. For example the map `[[0, 4], [2, 5]]` will take the set of layers `[0, 1, 2, 3,
    4]` and replace them with a module list containing `[0, 1, 2, 3, 2, 3, 4]`.
    """
    while hasattr(model, "model"):
        model = model.model
    # Some variants of the bert model nest the main model under the bert attribute.
    if hasattr(model, "bert"):
        model = model.bert

    model_type = None
    layers: nn.ModuleList = None
    if hasattr(model, "layers"):
        model_type = "llama"
        layers = model.layers
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        model_type = "bert"
        layers = model.encoder.layer
    elif hasattr(model, "h"):
        model_type = "falcon"
        layers = model.h
    if not model_type or not isinstance(layers, nn.ModuleList):
        raise ValueError(
            "Could not locate the layers attribute in the model. "
            "Expected Llama, Bert or Falcon compatible architectures."
        )

    new_layers = []
    for start, end in layer_map:
        for i in range(start, end):
            current_idx = len(new_layers)
            new_layers.append(clone_module(layers[i], share_weights=True))
            # This is a hack needed to work around the layer_idx introduced in HF transformers.
            for submodule in new_layers[-1].modules():
                if hasattr(submodule, "layer_idx"):
                    submodule.layer_idx = current_idx
    layers = nn.ModuleList(new_layers)
    if model_type == "llama":
        model.layers = layers
    elif model_type == "bert":
        model.encoder.layer = layers
    elif model_type == "falcon":
        model.h = layers
    else:
        raise ValueError("Unexpected model type, need to handle post-processing of layers.")
    if hasattr(model.config, "num_hidden_layers"):  # Common to Llama, Bert, Falcon.
        model.config.num_hidden_layers = len(new_layers)

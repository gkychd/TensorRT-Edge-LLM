# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
ONNX Export Module for LLM Models with Custom Attention Plugin

This module provides functionality to export different types of LLM models to ONNX format
with custom attention plugin integration. It supports standard models and EAGLE models.

ONNX Input Naming Conventions:
- inputs_embeds: Input embeddings for all models (replaces input_ids + image_embeds)
- deepstack_embeds: Deepstack visual embeddings for Qwen3VL and Qwen3Omni models (list of 3 tensors, each with shape (batch_size, seq_len, hidden_size))
- hidden_states_input: Renamed from hidden_states_from_base for ONNX export
- attention_pos_id: Renamed from position_ids for ONNX export

Model Loading Strategy:
- Standard models: Use AutoModelForCausalLM/AutoModelForImageTextToText detection
- EAGLE models: Load both base and draft models with weight copying

Embedding Export:
- All LLM models (both EAGLE base and regular): Export embedding.safetensors containing embedding layer weights
- Draft models only: Do not export embeddings (use base model embeddings)

Qwen3VL and Qwen3Omni Deepstack Processing:
- Deepstack embeddings are provided as 3 tensors with shape (batch_size, seq_len, hidden_size)
- Each tensor is directly added to hidden_states at specific decoder layers
- Simple element-wise addition for clean ONNX graph
"""

import json
import os
import shutil
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..quantization.quantization_utils import \
    enable_huggingface_checkpointing_patch

enable_huggingface_checkpointing_patch()

from ..chat_templates import (get_template_path, process_chat_template,
                              validate_chat_template)
from ..common import ONNX_OPSET_VERSION
from ..llm_models.layers.attention_plugin import \
    register_attention_plugin_onnx_symbolic_functions
from ..llm_models.layers.attention_trt import \
    register_trt_native_attention_onnx_symbolic_functions
from ..llm_models.layers.gather_nd import \
    register_gather_nd_onnx_symbolic_functions
from ..llm_models.layers.int4_gemm_plugin import (
    register_int4_gemm_plugin_onnx_symbolic_functions,
    replace_quant_linear_with_plugin)
from ..llm_models.layers.int4_moe_plugin import (
    is_moe_model, register_int4_moe_plugin_onnx_symbolic_functions,
    replace_moe_blocks_with_plugin)
from ..llm_models.layers.mamba_plugin import \
    register_mamba_plugin_onnx_symbolic_functions
from ..llm_models.model_utils import (is_gptq_model,
                                      is_incompatible_chat_template_model,
                                      load_eagle3_draft_model, load_llm_model,
                                      load_reduced_vocab_map)
from ..llm_models.models.llm_model import EdgeLLMHybridModelForCausalLM
from ..llm_models.models.llm_model_trtnative import (Eagle3DraftModelTRTNative,
                                                     EdgeLLMModelTRTNative)
from ..llm_models.models.qwen3_omni_talker import (
    create_qwen3_omni_dummy_inputs, export_qwen3_omni_submodel_to_onnx)
from .config_export import export_llm_config
from .onnx_utils_gai import export_onnx


def save_d2t_for_eagle3_draft(draft_model: nn.Module, output_dir: str) -> None:
    """Save d2t.safetensors for Eagle3 draft model."""
    from safetensors.torch import save_file

    d2t_tensor = draft_model.d2t
    # Convert to int32 and move to CPU if needed
    d2t_tensor_int32 = d2t_tensor.cpu().to(torch.int32)

    # Save as safetensors with key 'd2t'
    d2t_path = os.path.join(output_dir, "d2t.safetensors")
    save_file({"d2t": d2t_tensor_int32}, d2t_path)
    print(f"Saved d2t.safetensors to {output_dir}")


def save_embedding_table(base_model: nn.Module, output_dir: str) -> None:
    """Save embedding.safetensors for LLM models (both EAGLE base and regular models).
    
    Note: Draft models do not need embeddings as they use the base model's embeddings.
    """
    from safetensors.torch import save_file

    # Get the embedding layer from the model
    embed_tokens = base_model.embed_tokens
    embedding_weight = embed_tokens.weight.data.cpu()

    # Save as safetensors with key 'embedding'
    embedding_path = os.path.join(output_dir, "embedding.safetensors")
    save_file({"embedding": embedding_weight}, embedding_path)
    print(f"Saved embedding.safetensors to {output_dir}")


# ============================================================================
# Model-specific export hooks (extensible pattern)
# ============================================================================


def get_model_save_weights_hook(model_name: str):
    """
    Get weight saving function for each model type.
    
    Every model type has an explicit hook. Talker and CodePredictor have
    additional weights beyond the standard embedding table.
    """
    if model_name == "talker":

        def save_talker_weights(model, output_dir):
            save_embedding_table(model.transformer, output_dir)
            from ..llm_models.models.qwen3_omni_talker import \
                save_qwen3_omni_talker_projections
            save_qwen3_omni_talker_projections(model, output_dir)

        return save_talker_weights

    if model_name == "code_predictor":

        def save_code_predictor_weights(model, output_dir):
            from ..llm_models.models.qwen3_omni_talker import (
                save_qwen3_omni_code_predictor_embeddings,
                save_qwen3_omni_code_predictor_lm_heads)
            save_qwen3_omni_code_predictor_embeddings(model, output_dir)
            save_qwen3_omni_code_predictor_lm_heads(model, output_dir)
            # small_to_mtp_projection (TTS only — projects talker hidden to CP dimension)
            proj = getattr(model, 'small_to_mtp_projection', None)
            if proj is not None and not isinstance(proj, nn.Identity):
                from safetensors.torch import save_file
                save_file(
                    {
                        "weight": proj.weight.data.cpu().half(),
                        "bias": proj.bias.data.cpu().half()
                    },
                    os.path.join(output_dir,
                                 "small_to_mtp_projection.safetensors"),
                )
                print(
                    f"Saved small_to_mtp_projection.safetensors to {output_dir}"
                )

        return save_code_predictor_weights

    # Standard LLM / Thinker / EAGLE: only need embedding table
    def save_default_weights(model, output_dir):
        save_embedding_table(model, output_dir)

    return save_default_weights


def get_model_config_export_hook(model_name: str,
                                 model_dir: str = None,
                                 is_eagle_base: bool = False,
                                 trt_native_ops: bool = False):
    """
    Get config export function for each model type.
    
    Every model type has an explicit hook that returns a config dict.
    """
    if model_name == "talker":
        if not model_dir:
            raise ValueError("model_dir is required for talker config export")

        from transformers import AutoConfig

        from .config_export import (export_talker_config,
                                    export_tts_talker_config)

        def export_talker_config_hook(model_config):
            full_config = AutoConfig.from_pretrained(model_dir,
                                                     trust_remote_code=True)
            # Omni talker config has thinker_hidden_size; TTS does not
            has_thinker = hasattr(full_config, 'talker_config') and \
                          hasattr(full_config.talker_config, 'thinker_hidden_size')
            if has_thinker:
                return export_talker_config(full_config)
            else:
                return export_tts_talker_config(full_config)

        return export_talker_config_hook

    if model_name == "code_predictor":

        def export_code_predictor_config_hook(model_config):
            config = export_llm_config(model_config, 'llm', trt_native_ops)
            config["use_embeddings_input"] = True
            return config

        return export_code_predictor_config_hook

    if model_name == "thinker":

        def export_thinker_config_hook(model_config):
            config = export_llm_config(model_config, 'llm', trt_native_ops)
            if model_dir:
                from transformers import AutoConfig
                try:
                    full_config = AutoConfig.from_pretrained(
                        model_dir, trust_remote_code=True)
                except Exception:
                    full_config = None

                search_configs = [
                    getattr(full_config, 'thinker_config', None)
                    if full_config else None,
                    getattr(full_config, 'text_config', None)
                    if full_config else None,
                    full_config,
                    model_config,
                ]
                for field in [
                        "audio_token_id", "image_token_id", "video_token_id"
                ]:
                    for cfg in search_configs:
                        if cfg is None:
                            continue
                        val = getattr(cfg, field, None)
                        if val is not None:
                            config[field] = val
                            break
            return config

        return export_thinker_config_hook

    # Standard LLM / EAGLE base
    model_type = 'eagle3_base' if is_eagle_base else 'llm'

    def export_default_config_hook(model_config):
        return export_llm_config(model_config, model_type, trt_native_ops)

    return export_default_config_hook


def is_qwen3_omni_submodel(model_name: str) -> bool:
    """Check if model is a Qwen3-Omni submodel that needs special ONNX export."""
    return model_name in ["talker", "code_predictor"]


def create_dummy_inputs(model: nn.Module,
                        is_eagle_base: bool,
                        is_eagle_draft: bool,
                        fp8_kv_cache: bool = False) -> Dict[str, Any]:
    """
    Create dummy inputs for ONNX export.
    
    Args:
        model: The model to create inputs for
        is_eagle_base: Whether this is an EAGLE base model
        is_eagle_draft: Whether this is an EAGLE draft model
        fp8_kv_cache: Whether to use FP8 KV cache
        
    Returns:
        dict: Dictionary containing dummy inputs
    """
    # Use hardcoded values
    batch_size = 1
    seq_len = 2
    past_len = 2

    print(
        f"Creating dummy inputs with batch_size={batch_size}, seq_len={seq_len}, past_len={past_len}"
    )

    # Get model configuration
    model_config = model.config
    if model_config.model_type in ["qwen3_omni_thinker", "qwen3_asr"]:
        model_config = model_config.text_config

    hidden_size = model_config.hidden_size
    num_layers = model_config.num_hidden_layers
    num_heads = model_config.num_attention_heads
    num_kv_heads = model_config.num_key_value_heads
    # Use head_dim from config if available, otherwise calculate from hidden_size
    if hasattr(model_config, 'head_dim'):
        head_dim = model_config.head_dim
    else:
        head_dim = hidden_size // num_heads

    # Determine rotary dimension from partial_rotary_factor if provided
    partial_rotary_factor = getattr(model_config, 'partial_rotary_factor', 1.0)
    rotary_dim = int(head_dim * float(partial_rotary_factor))
    if rotary_dim <= 0 or rotary_dim > head_dim:
        rotary_dim = head_dim
    max_position_embeddings = model_config.max_position_embeddings

    device = next(model.parameters()).device

    # Create dummy past key values
    past_key_values = []
    for _ in range(num_layers):
        # Only FP16 KV Cache is supported for now. More precision will be supported in the future.
        past_key_value = torch.randn(batch_size,
                                     2,
                                     num_kv_heads,
                                     seq_len,
                                     head_dim,
                                     dtype=torch.float16,
                                     device=device)
        if fp8_kv_cache:
            past_key_value = past_key_value.to(torch.float8_e4m3fn)
        past_key_values.append(past_key_value)

    # Create last_token_ids
    if not is_eagle_base and not is_eagle_draft:
        last_token_ids = torch.full([batch_size, 1],
                                    seq_len - 1,
                                    dtype=torch.int64,
                                    device=device)
    else:
        # For EAGLE models, maintain batch dimension for proper GatherND support
        num_selected_tokens = 2
        last_token_ids = torch.full([batch_size, num_selected_tokens],
                                    seq_len - 1,
                                    dtype=torch.int64,
                                    device=device)

    # Create rope_rotary_cos_sin using rotary_dim
    rope_rotary_cos_sin = torch.randn(batch_size,
                                      max_position_embeddings,
                                      rotary_dim,
                                      dtype=torch.float32,
                                      device=device)

    # Create context_lengths
    context_lengths = torch.full([batch_size],
                                 past_len + seq_len,
                                 dtype=torch.int32,
                                 device=device)

    # Base inputs that all models need
    base_inputs = {
        'past_key_values': tuple(past_key_values),
        'last_token_ids': last_token_ids,
        'rope_rotary_cos_sin': rope_rotary_cos_sin,
        'context_lengths': context_lengths
    }

    # Create input_embeds for all models instead of input_ids
    inputs_embeds = torch.randn(batch_size,
                                seq_len,
                                hidden_size,
                                dtype=torch.float16,
                                device=device)
    base_inputs['inputs_embeds'] = inputs_embeds

    # For Qwen3VL and Qwen3OmniThinker, add deepstack visual embeds
    if model_config.model_type in ["qwen3_vl_text", "qwen3_omni_text"]:
        deepstack_visual_embeds = [
            torch.randn(batch_size,
                        seq_len,
                        hidden_size,
                        dtype=torch.float16,
                        device=device) for _ in range(3)
        ]
        base_inputs['deepstack_visual_embeds'] = deepstack_visual_embeds

    # Create position_ids and attention_mask for all models
    position_ids = torch.arange(seq_len, dtype=torch.int32,
                                device=device).unsqueeze(0).expand(
                                    batch_size, -1)
    attention_mask = torch.ones(batch_size,
                                seq_len,
                                seq_len + past_len,
                                dtype=torch.int32,
                                device=device)
    base_inputs['position_ids'] = position_ids
    base_inputs['attention_mask'] = attention_mask

    # kvcache_start_index is always required with shape [batch_size]
    base_inputs['kvcache_start_index'] = torch.zeros(batch_size,
                                                     dtype=torch.int32,
                                                     device=device)

    # Add EAGLE-specific inputs
    if is_eagle_draft:
        target_hidden_size = getattr(model_config, 'target_hidden_size',
                                     hidden_size)
        target_hidden_size = target_hidden_size * 3
        base_inputs['hidden_states_from_base'] = torch.randn(
            batch_size,
            seq_len,
            target_hidden_size,
            dtype=torch.float16,
            device=device)
        base_inputs['hidden_states_from_draft'] = torch.randn(
            batch_size,
            seq_len,
            hidden_size,
            dtype=torch.float16,
            device=device)

    return base_inputs


def replace_torch_quant_linear_with_int4_plugin(model: nn.Module) -> nn.Module:
    """
    Replace all TorchQuantLinear modules in a model with Int4GemmPluginModule.
        
    Args:
        model: PyTorch model containing TorchQuantLinear modules
        
    Returns:
        nn.Module: Model with TorchQuantLinear modules replaced by Int4GemmPluginModule
    """
    if is_gptq_model(model):
        print(
            "Detected GPTQ quantization, replacing quant linear with Int4GemmPluginModule"
        )
        register_int4_gemm_plugin_onnx_symbolic_functions()
        model = replace_quant_linear_with_plugin(model)
    return model


def create_hybrid_dummy_inputs(
        model: EdgeLLMHybridModelForCausalLM) -> Dict[str, Any]:
    """Create dummy inputs for hybrid Mamba+Attention ONNX export."""
    batch_size = 1
    seq_len = 2
    past_len = 2

    config = model.config
    hidden_size = config.hidden_size
    num_kv_heads = config.num_key_value_heads
    num_attn_heads = config.num_attention_heads

    if hasattr(config, 'head_dim'):
        head_dim = config.head_dim
    else:
        head_dim = hidden_size // num_attn_heads

    partial_rotary_factor = getattr(config, 'partial_rotary_factor', 1.0)
    rotary_dim = int(head_dim * float(partial_rotary_factor))
    if rotary_dim <= 0 or rotary_dim > head_dim:
        rotary_dim = head_dim
    max_position_embeddings = config.max_position_embeddings

    device = next(model.parameters()).device

    num_attn_layers = model.model.num_attention_layers
    num_mamba_layers = model.model.num_mamba_layers

    mamba_num_heads = config.mamba_num_heads
    mamba_head_dim = config.mamba_head_dim
    ssm_state_size = config.ssm_state_size

    past_key_values = []
    for _ in range(num_attn_layers):
        past_key_values.append(
            torch.randn(batch_size,
                        2,
                        num_kv_heads,
                        seq_len,
                        head_dim,
                        dtype=torch.float16,
                        device=device))

    # Conv states (only for mamba layers)
    conv_dim = config.mamba_num_heads * config.mamba_head_dim + 2 * config.n_groups * ssm_state_size
    conv_kernel = config.conv_kernel
    conv_states = []
    for _ in range(num_mamba_layers):
        conv_states.append(
            torch.zeros(batch_size,
                        conv_dim,
                        conv_kernel,
                        dtype=torch.float16,
                        device=device))

    # SSM states (only for mamba layers)
    ssm_states = []
    for _ in range(num_mamba_layers):
        ssm_states.append(
            torch.zeros(batch_size,
                        mamba_num_heads,
                        mamba_head_dim,
                        ssm_state_size,
                        dtype=torch.float16,
                        device=device))

    inputs_embeds = torch.randn(batch_size,
                                seq_len,
                                hidden_size,
                                dtype=torch.float16,
                                device=device)

    return {
        'inputs_embeds':
        inputs_embeds,
        'past_key_values':
        tuple(past_key_values),
        'conv_states':
        tuple(conv_states),
        'ssm_states':
        tuple(ssm_states),
        'rope_rotary_cos_sin':
        torch.randn(batch_size,
                    max_position_embeddings,
                    rotary_dim,
                    dtype=torch.float32,
                    device=device),
        'context_lengths':
        torch.full([batch_size],
                   past_len + seq_len,
                   dtype=torch.int32,
                   device=device),
        'last_token_ids':
        torch.full([batch_size, 1],
                   seq_len - 1,
                   dtype=torch.int64,
                   device=device),
        'kvcache_start_index':
        torch.zeros(batch_size, dtype=torch.int32, device=device),
    }


def export_hybrid_model_to_onnx(model: EdgeLLMHybridModelForCausalLM,
                                output_dir: str) -> None:
    """Export a hybrid Mamba+Attention model to ONNX."""
    print(f"Exporting hybrid model to ONNX format: {output_dir}")

    dummy_inputs = create_hybrid_dummy_inputs(model)
    model.eval()

    num_attn_layers = model.model.num_attention_layers
    num_mamba_layers = model.model.num_mamba_layers

    inputs = (
        dummy_inputs['inputs_embeds'],
        dummy_inputs['past_key_values'],
        dummy_inputs['conv_states'],
        dummy_inputs['ssm_states'],
        dummy_inputs['rope_rotary_cos_sin'],
        dummy_inputs['context_lengths'],
        dummy_inputs['last_token_ids'],
        dummy_inputs['kvcache_start_index'],
        None,  # position_ids
        None,  # attention_mask
    )

    input_names = (['inputs_embeds'] +
                   [f'past_key_values_{i}' for i in range(num_attn_layers)] +
                   [f'conv_state_{i}' for i in range(num_mamba_layers)] +
                   [f'ssm_state_{i}' for i in range(num_mamba_layers)] + [
                       'rope_rotary_cos_sin', 'context_lengths',
                       'last_token_ids', 'kvcache_start_index'
                   ])

    output_names = (
        ['logits'] +
        [f'present_key_values_{i}' for i in range(num_attn_layers)] +
        [f'present_conv_state_{i}' for i in range(num_mamba_layers)] +
        [f'present_ssm_state_{i}' for i in range(num_mamba_layers)])

    dynamic_axes = {
        'inputs_embeds': {
            0: 'batch_size',
            1: 'seq_len'
        },
        'rope_rotary_cos_sin': {
            0: 'rope_batch_size',
            1: 'max_position_embeddings'
        },
        'context_lengths': {
            0: 'batch_size'
        },
        'last_token_ids': {
            0: 'batch_size'
        },
        'kvcache_start_index': {
            0: 'kv_cache_start_batch_size'
        },
        'logits': {
            0: 'batch_size',
            1: 'num_tokens'
        },
    }
    for i in range(num_attn_layers):
        dynamic_axes[f'past_key_values_{i}'] = {0: 'batch_size', 3: 'past_len'}
        dynamic_axes[f'present_key_values_{i}'] = {
            0: 'batch_size',
            3: 'present_kv_cache_len'
        }
    for i in range(num_mamba_layers):
        dynamic_axes[f'conv_state_{i}'] = {0: 'batch_size'}
        dynamic_axes[f'present_conv_state_{i}'] = {0: 'batch_size'}
        dynamic_axes[f'ssm_state_{i}'] = {0: 'batch_size'}
        dynamic_axes[f'present_ssm_state_{i}'] = {0: 'batch_size'}

    register_attention_plugin_onnx_symbolic_functions()
    register_mamba_plugin_onnx_symbolic_functions()
    register_gather_nd_onnx_symbolic_functions()

    custom_opsets = {"trt_edgellm": ONNX_OPSET_VERSION}
    export_onnx(model,
                inputs,
                output_dir,
                input_names,
                output_names,
                dynamic_axes,
                custom_opsets=custom_opsets)


def export_model_to_onnx_with_trt_native_ops(model, output_dir: str) -> None:
    """
    Export the model to ONNX format with TensorRT native operations.
    
    Args:
        model: Model to export (EdgeLLMModelTRTNative or Eagle3DraftModelTRTNative)
        output_dir: Directory to save the exported ONNX model
    """
    assert isinstance(
        model, (EdgeLLMModelTRTNative, Eagle3DraftModelTRTNative)
    ), "Model must be an instance of EdgeLLMModelTRTNative or Eagle3DraftModelTRTNative"

    try:
        device = next(model.parameters()).device
        dummy_inputs, input_names, dynamic_axes, output_names = model.prepare_onnx_required_arguments(
            model.config, device)
        register_gather_nd_onnx_symbolic_functions()
        register_trt_native_attention_onnx_symbolic_functions()

        # Export to ONNX
        export_onnx(model, tuple(dummy_inputs), output_dir, input_names,
                    output_names, dynamic_axes)

    except Exception as e:
        raise RuntimeError(f"Failed to export model to ONNX: {str(e)}")


def export_model_to_onnx(model: nn.Module,
                         output_dir: str,
                         is_eagle_base: bool,
                         is_eagle_draft: bool,
                         fp8_kv_cache: bool = False) -> None:
    """
    Export the model to ONNX format.
    
    Args:
        model: The model to export
        output_dir: Directory to save the ONNX model
        is_eagle_base: Whether this is an EAGLE base model
        is_eagle_draft: Whether this is an EAGLE draft model
        fp8_kv_cache: Whether to use FP8 KV cache
    """
    print(f"Exporting model to ONNX format: {output_dir}")

    dummy_inputs = create_dummy_inputs(model, is_eagle_base, is_eagle_draft,
                                       fp8_kv_cache)

    # Auto-detect if model should export hidden_states output
    # - EAGLE base: needs hidden_states for draft model input
    # - EAGLE draft: needs hidden_states for speculative decoding verification
    # - Qwen3-Omni Thinker: always export hidden_states (runtime decides whether to use it)
    model_type = getattr(model.config, 'model_type', '')
    needs_hidden_states = is_eagle_base or is_eagle_draft or (
        model_type == "qwen3_omni_text")

    try:
        # Set model to evaluation mode
        model.eval()

        # Get model configuration for dynamic shapes
        model_config = model.config
        if model_config.model_type in ["qwen3_omni_thinker", "qwen3_asr"]:
            model_config = model_config.text_config
        num_layers = model_config.num_hidden_layers

        # Prepare inputs - order must match model forward signature
        # For LLM: inputs_embeds, past_key_values, rope_rotary_cos_sin, context_lengths, last_token_ids, kvcache_start_index, position_ids, attention_mask, deepstack_visual_embeds
        # For Draft: inputs_embeds, past_key_values, rope_rotary_cos_sin, context_lengths, last_token_ids, kvcache_start_index, hidden_states_from_base, hidden_states_from_draft, position_ids, attention_mask

        base_inputs = [
            dummy_inputs['inputs_embeds'],
            dummy_inputs['past_key_values'],
            dummy_inputs['rope_rotary_cos_sin'],
            dummy_inputs['context_lengths'],
            dummy_inputs['last_token_ids'],
            dummy_inputs['kvcache_start_index'],
        ]

        if is_eagle_draft:
            base_inputs.extend([
                dummy_inputs['hidden_states_from_base'],
                dummy_inputs['hidden_states_from_draft'],
                dummy_inputs['position_ids'], dummy_inputs['attention_mask']
            ])
        elif is_eagle_base:
            base_inputs.extend(
                [dummy_inputs['position_ids'], dummy_inputs['attention_mask']])
        else:
            # Standard models pass None for position_ids and attention_mask
            base_inputs.extend([None, None])

        # For Qwen3VL and Qwen3Omni Thinker, add deepstack visual embeds
        require_deepstack_embeds = model_config.model_type in [
            "qwen3_vl_text", "qwen3_omni_text"
        ]
        if require_deepstack_embeds:
            base_inputs.extend([dummy_inputs['deepstack_visual_embeds']])

        inputs = tuple(base_inputs)

        # Create input names
        input_names = (['inputs_embeds'] +
                       [f'past_key_values_{i}' for i in range(num_layers)] + [
                           'rope_rotary_cos_sin', 'context_lengths',
                           'last_token_ids', 'kvcache_start_index'
                       ])

        if is_eagle_draft:
            input_names += [
                'hidden_states_input', 'hidden_states_from_draft',
                'attention_pos_id', 'attention_mask'
            ]
        elif is_eagle_base:
            input_names += ['attention_pos_id', 'attention_mask']

        if require_deepstack_embeds:
            input_names += [f'deepstack_embeds_{i}' for i in range(3)]

        # Create output names
        # EAGLE base, EAGLE draft, and Qwen3-Omni Thinker output hidden_states
        # Standard LLMs only output logits and present_key_values
        if needs_hidden_states:
            output_names = ['logits', 'hidden_states'] + \
                           [f'present_key_values_{i}' for i in range(num_layers)]
        else:
            output_names = ['logits'] + \
                           [f'present_key_values_{i}' for i in range(num_layers)]

        # Create dynamic axes
        dynamic_axes = {
            **{
                f"past_key_values_{i}": {
                    0: "batch_size",
                    3: "past_len"
                }
                for i in range(num_layers)
            },
            **{
                f"present_key_values_{i}": {
                    0: "batch_size",
                    3: "present_kv_cache_len"
                }
                for i in range(num_layers)
            },
            "inputs_embeds": {
                0: "batch_size",
                1: "seq_len"
            },
            "rope_rotary_cos_sin": {
                0: "rope_batch_size",
                1: "max_position_embeddings"
            },
            "context_lengths": {
                0: "batch_size"
            },
            "last_token_ids": {
                0: "batch_size",
                1: "num_selected_tokens"
            } if (is_eagle_base or is_eagle_draft) else {
                0: "batch_size"
            },
            "kvcache_start_index": {
                0: "kv_cache_start_batch_size"
            },
            "logits": {
                0:
                "batch_size",
                1:
                "num_selected_tokens" if
                (is_eagle_base or is_eagle_draft) else "num_tokens"
            },
        }

        if is_eagle_draft:
            dynamic_axes.update({
                "hidden_states_input": {
                    0: "batch_size",
                    1: "seq_len"
                },
                "hidden_states_from_draft": {
                    0: "batch_size",
                    1: "seq_len"
                },
            })

        # EAGLE base, EAGLE draft, and Qwen3-Omni Thinker output hidden_states
        if needs_hidden_states:
            dynamic_axes.update({
                "hidden_states": {
                    0: "batch_size",
                    1: "seq_len"
                },
            })

        if is_eagle_base or is_eagle_draft:
            dynamic_axes.update({
                "attention_pos_id": {
                    0: "batch_size",
                    1: "q_len"
                },
                "attention_mask": {
                    0: "batch_size",
                    1: "q_len",
                    2: "q_len_padded"
                },
            })

        if require_deepstack_embeds:
            dynamic_axes.update({
                **{
                    f"deepstack_embeds_{i}": {
                        0: "batch_size",
                        1: "seq_len"
                    }
                    for i in range(3)
                },
            })

        # Register ONNX symbolic functions
        register_attention_plugin_onnx_symbolic_functions()
        register_gather_nd_onnx_symbolic_functions()

        # Export to ONNX
        export_onnx(model, inputs, output_dir, input_names, output_names,
                    dynamic_axes)

    except Exception as e:
        raise RuntimeError(f"Failed to export model to ONNX: {str(e)}")


def export_llm_model(model_dir: str,
                     output_dir: str,
                     device: str = "cuda",
                     is_eagle_base: bool = False,
                     reduced_vocab_dir: Optional[str] = None,
                     chat_template_path: Optional[str] = None,
                     fp8_kv_cache: bool = False,
                     trt_native_ops: bool = False,
                     export_models: Optional[str] = None) -> None:
    """
    Export a language model to ONNX format with custom attention plugin.
    
    This is the main entry point for exporting standard LLM models and EAGLE base models
    to ONNX format with TensorRT Edge-LLM optimizations.
    
    Args:
        model_dir: Directory containing the HuggingFace model
        output_dir: Directory to save the exported ONNX model
        device: Device to load the model on ("cpu", "cuda", or "cuda:0", "cuda:1", etc.)
        is_eagle_base: Whether the model is an EAGLE3 base model (vs standard LLM)
        reduced_vocab_dir: Directory containing vocab_map.safetensors for vocabulary reduction (optional)
        chat_template_path: Path to chat template JSON file. When provided, this template is validated and used instead of inferring from the model (optional)
        fp8_kv_cache: Whether to use FP8 KV cache
        trt_native_ops: Whether to use TensorRT native operations instead of plugin
        export_models: Comma-separated list of models to export for Qwen3-Omni (e.g., "thinker,talker"). Default: export all models
    """
    start_time = time.time()

    # Parse export_models filter (for selective multi-model exports like Qwen3-Omni)
    export_models_set = None
    if export_models:
        export_models_set = set(m.strip() for m in export_models.split(','))
        print(f"Export filter: only exporting {export_models_set}")

    if is_eagle_base:
        print(f"Exporting EAGLE3 base model to ONNX format")
    else:
        print(f"Exporting standard model to ONNX format")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load reduced vocabulary map if provided
    reduced_vocab_size = None
    vocab_map = None
    if reduced_vocab_dir is not None:
        print(f"Loading reduced vocabulary from {reduced_vocab_dir}")
        reduced_vocab_size, vocab_map = load_reduced_vocab_map(
            reduced_vocab_dir, device)

    # Load model(s)
    # Always returns dict for uniform processing: {"model_name": model, ...}
    models_or_dict, tokenizer, processor = load_llm_model(
        model_dir,
        dtype='fp16',
        device=device,
        is_eagle_base=is_eagle_base,
        reduced_vocab_size=reduced_vocab_size,
        vocab_map=vocab_map,
        trt_native_ops=trt_native_ops)

    # Normalize to dict (single model wrapped as {"model": model})
    models_dict = models_or_dict if isinstance(models_or_dict, dict) else {
        "model": models_or_dict
    }
    is_multi_model = len(models_dict) > 1

    # ========== Standard Export Flow ==========
    # Export each model with unified pipeline
    for model_name, model in models_dict.items():
        # Filter check
        if export_models_set is not None and model_name not in export_models_set:
            print(f"Skipping {model_name}")
            continue

        print(f"\n=== Exporting {model_name} ===")
        model_output_dir = os.path.join(
            output_dir, model_name) if is_multi_model else output_dir

        if is_moe_model(model):
            print(
                "Detected MoE model, replacing MoE blocks with Int4MoePlugin")
            register_int4_moe_plugin_onnx_symbolic_functions()
            model = replace_moe_blocks_with_plugin(model)

        # Step 1: Apply model modifications
        model = replace_torch_quant_linear_with_int4_plugin(model)

        # Step 2: Export ONNX
        if trt_native_ops:
            export_model_to_onnx_with_trt_native_ops(model, model_output_dir)
        elif isinstance(model, EdgeLLMHybridModelForCausalLM):
            export_hybrid_model_to_onnx(model, model_output_dir)
        elif is_qwen3_omni_submodel(model_name):
            dummy_inputs = create_qwen3_omni_dummy_inputs(
                model, model_name, fp8_kv_cache)
            export_qwen3_omni_submodel_to_onnx(model, dummy_inputs,
                                               model_output_dir, model_name)
        else:
            export_model_to_onnx(model, model_output_dir, is_eagle_base, False,
                                 fp8_kv_cache)

        # Step 3: Export config
        if isinstance(model, EdgeLLMHybridModelForCausalLM):
            model_config = export_llm_config(model.config, 'hybrid_mamba',
                                             trt_native_ops)
        else:
            config_hook = get_model_config_export_hook(model_name, model_dir,
                                                       is_eagle_base,
                                                       trt_native_ops)
            model_config = config_hook(model.config)

        if reduced_vocab_size is not None:
            model_config['reduced_vocab_size'] = reduced_vocab_size

        with open(os.path.join(model_output_dir, "config.json"), 'w') as f:
            json.dump(model_config, f, indent=2)
        print(f"Config saved to {model_output_dir}")

        # Step 4: Save weights
        weights_hook = get_model_save_weights_hook(model_name)
        weights_hook(model, model_output_dir)

    # ========== Save Shared Resources ==========

    # Determine tokenizer save location
    # Omni (has thinker): save to thinker subdirectory (where llm_build looks)
    # TTS / single model: save to top-level
    if "thinker" in models_dict and is_multi_model:
        tokenizer_save_dir = os.path.join(output_dir, "thinker")
    else:
        tokenizer_save_dir = output_dir

    # Save tokenizer files
    tokenizer.save_pretrained(tokenizer_save_dir)
    print(f"Tokenizer saved to {tokenizer_save_dir}")

    # Save processor files if available
    if processor is not None:
        processor.save_pretrained(tokenizer_save_dir)
        print(f"Processor saved to {tokenizer_save_dir}")

    # Check if model requires explicit chat template
    is_incompatible, incompatible_model_type = is_incompatible_chat_template_model(
        model_dir)

    # Determine chat template source
    if chat_template_path is not None:
        # User provided a chat template
        template_source = chat_template_path
    elif is_incompatible:
        # Use template from chat_templates/templates/
        template_source = get_template_path(incompatible_model_type)
        if template_source is None:
            raise ValueError(
                f"Model '{incompatible_model_type}' requires the --chat_template flag.\n"
                f"This model type does not have a compatible chat template that can be "
                f"automatically extracted from its tokenizer, and no template is available.\n"
                f"Please provide a chat template JSON file using: --chat_template /path/to/template.json\n"
                f"See docs/source/user_guide/format/chat-template-format.md for the required format."
            )
    else:
        template_source = None

    # Handle chat template (save to tokenizer location)
    if template_source is not None:
        # Validate and copy the template
        print(f"Using chat template from: {template_source}")
        validate_chat_template(template_source)
        output_template_path = os.path.join(tokenizer_save_dir,
                                            "processed_chat_template.json")
        shutil.copy2(template_source, output_template_path)
        print(f"Chat template saved to {output_template_path}")
    else:
        # Generate chat template from model
        process_chat_template(model_dir, tokenizer_save_dir)

    # Copy vocab_map.safetensors to output directory if reduced_vocab_dir is provided
    if reduced_vocab_dir is not None:
        vocab_map_src = os.path.join(reduced_vocab_dir,
                                     "vocab_map.safetensors")
        vocab_map_dst = os.path.join(output_dir, "vocab_map.safetensors")
        if os.path.exists(vocab_map_src):
            shutil.copy2(vocab_map_src, vocab_map_dst)
            print(f"Copied vocab_map.safetensors to {output_dir}")
        else:
            print(
                f"Warning: vocab_map.safetensors not found in {reduced_vocab_dir}"
            )

    end_time = time.time()
    print(
        f"Export completed successfully in {end_time - start_time}s. Files saved to: {output_dir}"
    )


def export_draft_model(draft_model_dir: str,
                       output_dir: str,
                       base_model_dir: Optional[str] = None,
                       device: str = "cuda",
                       trt_native_ops: bool = False) -> None:
    """
    Export an EAGLE draft model to ONNX format.
    
    This is the main entry point for exporting EAGLE draft models to ONNX format.
    The draft model requires a base model for weight copying.
    
    Args:
        draft_model_dir: Directory containing the EAGLE draft model
        output_dir: Directory to save the exported ONNX model
        base_model_dir: Directory containing the base model (for weight copying)
        device: Device to load the model on ("cpu", "cuda", or "cuda:0", "cuda:1", etc.)
        trt_native_ops: Whether to use TensorRT native operations instead of plugin
    """
    start_time = time.time()

    if trt_native_ops:
        print("Exporting EAGLE3 draft model with TensorRT native operations")
    else:
        print("Exporting EAGLE3 draft model with custom attention plugin")

    # Create subdirectories
    os.makedirs(output_dir, exist_ok=True)

    # Load draft model with base model for weight copying
    print(f"Loading draft model from {draft_model_dir}")
    draft_model = load_eagle3_draft_model(draft_model_dir, base_model_dir,
                                          'fp16', device, trt_native_ops)

    draft_model = replace_torch_quant_linear_with_int4_plugin(draft_model)

    # Export draft model
    print(f"Exporting draft model to {output_dir}")
    if trt_native_ops:
        export_model_to_onnx_with_trt_native_ops(draft_model, output_dir)
    else:
        export_model_to_onnx(draft_model,
                             output_dir,
                             is_eagle_base=False,
                             is_eagle_draft=True,
                             fp8_kv_cache=False)

    # Save draft model configuration
    draft_config = export_llm_config(draft_model.config, 'eagle_draft',
                                     trt_native_ops)
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(draft_config, f, indent=2)
    print(f"Draft model configuration saved to {config_path}")

    # Save d2t mapping
    save_d2t_for_eagle3_draft(draft_model, output_dir)

    draft_end_time = time.time()
    print(
        f"Complete draft model export completed successfully in {draft_end_time - start_time}s!"
    )
    print(f"Draft model saved to: {output_dir}")

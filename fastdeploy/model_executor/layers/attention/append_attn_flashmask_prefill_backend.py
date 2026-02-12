"""
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
"""

from __future__ import annotations

import os
import numpy as np
from typing import TYPE_CHECKING

import paddle
import paddle.nn.functional as F
try:
    import paddlefleet  # type: ignore
except ModuleNotFoundError:
    paddlefleet = None

from fastdeploy.model_executor.layers.attention.append_attn_backend import AppendAttentionBackend
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.attention.ops import (
    get_block_shape_and_split_kv_block,
    gqa_rope_write_cache,
    init_signal_layerwise,
    pre_cache_len_concat,
)
from fastdeploy.platforms import current_platform
from fastdeploy.utils import console_logger as logger

if TYPE_CHECKING:
    from fastdeploy.model_executor.forward_meta import ForwardMeta

import tencap as tc
FD_DO_DEBUG_CAPTURE = os.getenv("FD_DO_DEBUG_CAPTURE", "0") == "1"

class AppendAttentionFlashMaskPrefillBackend(AppendAttentionBackend):
    """
    Experimental attention backend.

    - Prefill-only (first prefill) path: uses Paddle `flashmask_attention` v3, but still
      writes FastDeploy paged KV cache via `gqa_rope_write_cache` for subsequent decode.
    - Decode fallback path: uses Paddle flash attention for pure decode.

    Env knobs:
    - FD_USE_PADDLE_FLASHMASK_PREFILL=1: enable flashmask prefill path (when eligible)
    - FD_STRICT_PURE_PREFILL_DECODE=1: assert effective batch size==1 and pure prefill/decode only
    """

    _debug = os.getenv("FD_DEBUG_ATTN_BACKEND", "0").lower() in ("1", "true")
    _break_on_entry = os.getenv("FD_BREAK_ON_ATTN_BACKEND", "0").lower() in ("1", "true")
    _break_fired = False

    def forward_mixed(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        qkv: paddle.Tensor,
        compressed_kv: paddle.Tensor,
        k_pe: paddle.Tensor,
        layer: Attention,
        forward_meta: ForwardMeta,
    ) -> paddle.Tensor:
        strict_pure_prefill_decode = os.getenv("FD_STRICT_PURE_PREFILL_DECODE", "0").lower() in ("1", "true")
        use_paddle_flashmask_prefill = os.getenv("FD_USE_PADDLE_FLASHMASK_PREFILL", "0").lower() in ("1", "true")
        # rr_attention uses the same flashmask prefill path; enabling it implicitly requests that path.
        if self.enable_rr_attention:
            use_paddle_flashmask_prefill = True

        if self._break_on_entry and (not self._break_fired) and layer.layer_id == 0:
            type(self)._break_fired = True
            raise RuntimeError("FD_BREAK_ON_ATTN_BACKEND: hit AppendAttentionFlashMaskPrefillBackend.forward_mixed")

        if self._debug and layer.layer_id == 0:
            logger.info(
                "[AppendAttentionFlashMaskPrefillBackend] enter forward_mixed "
                f"(pid={os.getpid()}, strict={strict_pure_prefill_decode}, "
                f"use_flashmask={use_paddle_flashmask_prefill}, "
                f"step_use_cudagraph={getattr(forward_meta, 'step_use_cudagraph', None)}, "
                f"qkv_dtype={getattr(qkv, 'dtype', None)})"
            )
        if not (strict_pure_prefill_decode or use_paddle_flashmask_prefill):
            return super().forward_mixed(q, k, v, qkv, compressed_kv, k_pe, layer, forward_meta)

        metadata = self.attention_metadata
        sliding_window = layer.sliding_window

        if self.rope_3d:
            assert len(forward_meta.rotary_embs.shape) == 6
        else:
            assert len(forward_meta.rotary_embs.shape) == 5
            if layer.use_neox_rotary_style:
                assert forward_meta.rotary_embs.shape[0:4] == [2, 1, self.max_seq_len, 1]
                assert forward_meta.rotary_embs.shape[4] in [128, 32]

        if self.pd_disaggregation_mode == "per_query":
            metadata.kv_signal_data_list[layer.layer_id] = init_signal_layerwise(
                metadata.kv_signal_metadata,
                layer.layer_id + self.start_layer_index,
            )

        # Strict checks and flashmask path both rely on max_len_tensor_cpu computed below.
        if layer.layer_id == 0:
            get_block_shape_and_split_kv_block(
                forward_meta.seq_lens_encoder,
                forward_meta.seq_lens_decoder,
                forward_meta.seq_lens_this_time,
                forward_meta.decoder_batch_ids,
                forward_meta.decoder_tile_ids_per_batch,
                forward_meta.decoder_num_blocks_cpu,
                forward_meta.decoder_num_blocks_device,
                forward_meta.decoder_chunk_size_device,
                forward_meta.max_len_tensor_cpu,
                forward_meta.encoder_batch_ids,
                forward_meta.encoder_tile_ids_per_batch,
                forward_meta.encoder_num_blocks_x_cpu,
                forward_meta.kv_batch_ids,
                forward_meta.kv_tile_ids_per_batch,
                forward_meta.kv_num_blocks_x_cpu,
                self.encoder_block_shape_q,
                self.decoder_block_shape_q,
                self.group_size,
                self.block_size,
            )

        if strict_pure_prefill_decode and layer.layer_id == 0:
            if getattr(self.fd_config.cache_config, "enable_chunked_prefill", False):
                raise AssertionError(
                    "FD_STRICT_PURE_PREFILL_DECODE=1 requires chunked prefill disabled "
                    "(set enable_chunked_prefill=False or export FD_DISABLE_CHUNKED_PREFILL=1)."
                )

            max_len_tensor_cpu = forward_meta.max_len_tensor_cpu
            max_enc_len_this_time = int(max_len_tensor_cpu[1].item())
            max_dec_len_this_time = int(max_len_tensor_cpu[2].item())
            max_just_dec_len_this_time = int(max_len_tensor_cpu[4].item())

            # Avoid any GPU->CPU sync/copy here: during CUDA Graph capture, `.cpu()`/`.numpy()`/`.item()`
            # on GPU tensors will trigger `cudaErrorStreamCaptureImplicit`.
            # For strict mode, require the scheduler/engine to be configured with batch size 1.
            seq_lens_this_time = forward_meta.seq_lens_this_time
            if len(seq_lens_this_time.shape) not in (1, 2):
                raise AssertionError(
                    "FD_STRICT_PURE_PREFILL_DECODE=1 expects seq_lens_this_time to be rank-1/2 tensor, "
                    f"got shape={list(seq_lens_this_time.shape)}."
                )
            effective_batch_size = int(seq_lens_this_time.shape[0])
            if effective_batch_size != 1:
                raise AssertionError(
                    "FD_STRICT_PURE_PREFILL_DECODE=1 requires effective batch size=1. "
                    f"Got seq_lens_this_time.shape[0]={effective_batch_size}. "
                    "Please configure the engine with max_num_seqs=1 (e.g. LLM(..., max_num_seqs=1) "
                    "or CLI --max_num_seqs 1)."
                )

            is_pure_prefill = max_enc_len_this_time > 0 and max_dec_len_this_time == 0 and max_just_dec_len_this_time == 0
            is_pure_decode = max_enc_len_this_time == 0 and max_just_dec_len_this_time > 0
            if not (is_pure_prefill or is_pure_decode):
                raise AssertionError(
                    "FD_STRICT_PURE_PREFILL_DECODE=1 requires pure prefill (first prefill) or pure decode only. "
                    f"Got max_enc_len_this_time={max_enc_len_this_time}, "
                    f"max_dec_len_this_time={max_dec_len_this_time}, "
                    f"max_just_dec_len_this_time={max_just_dec_len_this_time}."
                )

        if use_paddle_flashmask_prefill and current_platform.is_cuda():
            # This experimental path must be CUDA Graph capture-safe (no GPU->CPU sync/copy).
            # For now, only support the strict single-sequence case.
            if not strict_pure_prefill_decode:
                if self._debug and layer.layer_id == 0:
                    logger.info(
                        "[AppendAttentionFlashMaskPrefillBackend] flashmask prefill requested but strict mode is off; "
                        "falling back to AppendAttentionBackend."
                    )
                return super().forward_mixed(q, k, v, qkv, compressed_kv, k_pe, layer, forward_meta)

            cache_quant_type_str = getattr(layer, "cache_quant_type_str", "none")
            if cache_quant_type_str == "none":
                cache_k = forward_meta.caches[2 * layer.layer_id]
                cache_v = forward_meta.caches[2 * layer.layer_id + 1]
            else:
                cache_k = None
                cache_v = None

            try:
                fa_version = paddle.base.framework.get_flags(["FLAGS_flash_attn_version"])["FLAGS_flash_attn_version"]
            except Exception:
                fa_version = None
            try:
                cudnn_deterministic = paddle.get_flags(["FLAGS_cudnn_deterministic"])["FLAGS_cudnn_deterministic"]
            except Exception:
                cudnn_deterministic = False

            max_len_tensor_cpu = forward_meta.max_len_tensor_cpu
            max_len_this_time = int(max_len_tensor_cpu[0].item())
            max_enc_len_this_time = int(max_len_tensor_cpu[1].item())
            max_dec_len_this_time = int(max_len_tensor_cpu[2].item())
            max_just_dec_len_this_time = int(max_len_tensor_cpu[4].item())

            eligible_flashmask_prefill = (
                fa_version >= 2
                and not (cudnn_deterministic and self.head_dim > 128)
                and max_enc_len_this_time > 0
                and max_just_dec_len_this_time == 0
                and max_dec_len_this_time == 0
                and cache_k is not None
                and cache_v is not None
                and qkv.dtype == paddle.bfloat16
                and forward_meta.attn_mask is None
                and forward_meta.attn_mask_offsets is None
                and sliding_window == 0
            )

            if self._debug and layer.layer_id == 0:
                logger.info(
                    "[AppendAttentionFlashMaskPrefillBackend] flashmask prefill eligibility: "
                    f"{eligible_flashmask_prefill} "
                    f"(fa_version={fa_version}, cudnn_deterministic={cudnn_deterministic}, "
                    f"max_enc_len_this_time={max_enc_len_this_time}, "
                    f"max_dec_len_this_time={max_dec_len_this_time}, "
                    f"max_just_dec_len_this_time={max_just_dec_len_this_time}, "
                    f"qkv_dtype={qkv.dtype}, "
                    f"attn_mask_is_none={forward_meta.attn_mask is None}, "
                    f"attn_mask_offsets_is_none={forward_meta.attn_mask_offsets is None}, "
                    f"sliding_window={sliding_window})"
                )

            if eligible_flashmask_prefill:
                if self._debug and layer.layer_id == 0:
                    if self.enable_rr_attention:
                        logger.info(
                            "[AppendAttentionFlashMaskPrefillBackend] using paddlefleet rr_attention for first prefill "
                            f"(threshold={self.rr_attention_threshold}, stride={self.rr_attention_stride})."
                        )
                    else:
                        logger.info(
                            "[AppendAttentionFlashMaskPrefillBackend] using Paddle flashmask_attention for first prefill."
                        )
                (
                    attn_cu_seqlens_k,
                    pre_cache_batch_ids,
                    pre_cache_tile_ids_per_batch,
                    pre_cache_num_blocks_cpu,
                    kv_token_num_cpu,
                ) = pre_cache_len_concat(
                    forward_meta.seq_lens_encoder,
                    forward_meta.seq_lens_decoder,
                    forward_meta.seq_lens_this_time,
                    max_dec_len_this_time,
                    self.block_size,
                )
                kv_token_num = int(kv_token_num_cpu[0].item())

                q_packed, k_packed, v_packed, _ = gqa_rope_write_cache(
                    qkv,
                    cache_k,
                    cache_v,
                    forward_meta.cu_seqlens_q,
                    attn_cu_seqlens_k,
                    forward_meta.rotary_embs,
                    forward_meta.seq_lens_this_time,
                    forward_meta.seq_lens_encoder,
                    forward_meta.seq_lens_decoder,
                    forward_meta.batch_id_per_token,
                    forward_meta.block_tables,
                    forward_meta.kv_batch_ids,
                    forward_meta.kv_tile_ids_per_batch,
                    forward_meta.kv_num_blocks_x_cpu,
                    pre_cache_batch_ids,
                    pre_cache_tile_ids_per_batch,
                    pre_cache_num_blocks_cpu,
                    getattr(layer, "q_norm_weight", None),
                    getattr(layer, "k_norm_weight", None),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    metadata.kv_signal_data_list[layer.layer_id],
                    kv_token_num=kv_token_num,
                    max_seq_len=self.max_seq_len,
                    rms_norm_eps=getattr(layer, "rms_norm_eps", 1e-6),
                    use_neox_rotary_style=layer.use_neox_rotary_style,
                    cache_quant_type="none",
                    rope_3d=self.rope_3d,
                )

                token_num = int(q_packed.shape[0])
                if int(k_packed.shape[0]) != token_num or int(v_packed.shape[0]) != token_num:
                    raise NotImplementedError("flashmask prefill path only supports first prefill (no KV history).")

                if token_num == 0:
                    return paddle.empty([0, self.num_heads * self.head_dim], dtype=qkv.dtype)

                if token_num > max_len_this_time:
                    raise RuntimeError(
                        "flashmask prefill expects token_num <= max_len_this_time, "
                        f"got token_num={token_num}, max_len_this_time={max_len_this_time}."
                    )

                q_dense = paddle.zeros([1, max_len_this_time, self.num_heads, self.head_dim], dtype=qkv.dtype)
                k_dense = paddle.zeros([1, max_len_this_time, self.kv_num_heads, self.head_dim], dtype=qkv.dtype)
                v_dense = paddle.zeros([1, max_len_this_time, self.kv_num_heads, self.head_dim], dtype=qkv.dtype)

                q_dense[0, :token_num] = q_packed
                k_dense[0, :token_num] = k_packed
                v_dense[0, :token_num] = v_packed

                if (not forward_meta.is_dummy_or_profile_run) and FD_DO_DEBUG_CAPTURE:
                    with tc.scope("fd_results"):
                        q_dense_np = q_packed.unsqueeze(0).astype("float32").numpy()
                        k_dense_np = k_packed.unsqueeze(0).astype("float32").numpy()
                        rotary_emb_np = forward_meta.rotary_embs.astype("float32").numpy()
                        tc.dump_np(q_dense_np, name="q_rope")
                        tc.dump_np(k_dense_np, name="k_rope")
                        tc.dump_np(rotary_emb_np, name="rot_emb")

                startend_tensor = paddle.full([1, 1, max_len_this_time, 1], max_len_this_time, dtype="int32")
                if token_num < max_len_this_time:
                    startend_tensor[:, :, token_num:, :] = 0
                startend_tensor = startend_tensor.to(q_dense.place)
                
                if self.enable_rr_attention:
                    if paddlefleet is None or not hasattr(getattr(paddlefleet, "ops", None), "rr_attention"):
                        raise RuntimeError(
                            "enable_rr_attention=True requires `paddlefleet` with `paddlefleet.ops.rr_attention` "
                            "available. Please install/enable paddlefleet or disable rr attention."
                        )
                    if self.head_dim != 128:
                        raise RuntimeError(
                            "enable_rr_attention=True requires head_dim==128 for the current rr_attention "
                            f"implementation, but got head_dim={self.head_dim}."
                        )
                    out_dense = paddlefleet.ops.rr_attention(
                        q_dense,
                        k_dense,
                        v_dense,
                        startend_tensor,
                        dropout=0.0,
                        causal=self.causal,
                        training=True,
                        threshold=self.rr_attention_threshold,
                        stride=self.rr_attention_stride,
                    )
                else:
                    out_dense = F.flashmask_attention(
                        q_dense,
                        k_dense,
                        v_dense,
                        startend_tensor,
                        dropout=0.0,
                        causal=self.causal,
                        training=True,
                    )

                return out_dense[0, :token_num].reshape([token_num, self.num_heads * self.head_dim])

        if self._debug and layer.layer_id == 0 and use_paddle_flashmask_prefill and current_platform.is_cuda():
            logger.info(
                "[AppendAttentionFlashMaskPrefillBackend] flashmask prefill path not taken; "
                "falling back to AppendAttentionBackend."
            )
        # return super().forward_mixed(q, k, v, qkv, compressed_kv, k_pe, layer, forward_meta)

        # Decode fallback: avoid super().forward_mixed and use flash attention.
        max_len_tensor_cpu = forward_meta.max_len_tensor_cpu
        max_enc_len_this_time = int(max_len_tensor_cpu[1].item())
        max_just_dec_len_this_time = int(max_len_tensor_cpu[4].item())
        is_pure_decode = max_enc_len_this_time == 0 and max_just_dec_len_this_time > 0
        if not is_pure_decode:
            raise RuntimeError(
                "AppendAttentionFlashMaskPrefillBackend decode fallback only supports pure decode. "
                f"max_enc_len_this_time={max_enc_len_this_time}, max_just_dec_len_this_time={max_just_dec_len_this_time}"
            )
        #     return super().forward_mixed(q, k, v, qkv, compressed_kv, k_pe, layer, forward_meta)

        cache_quant_type_str = getattr(layer, "cache_quant_type_str", "none")
        if cache_quant_type_str not in ("none", "cache_int8", "cache_fp8", "cache_int4_zp"):
            if self._debug and layer.layer_id == 0:
                logger.info(
                    "[AppendAttentionFlashMaskPrefillBackend] decode math fallback does not support "
                    f"cache_quant_type={cache_quant_type_str}; fallback to AppendAttentionBackend."
                )
            return super().forward_mixed(q, k, v, qkv, compressed_kv, k_pe, layer, forward_meta)

        cache_k = forward_meta.caches[2 * layer.layer_id]
        cache_v = forward_meta.caches[2 * layer.layer_id + 1]
        cache_k_quant_scales = getattr(layer, "cache_k_scale", None)
        cache_v_quant_scales = getattr(layer, "cache_v_scale", None)

        # gqa_rope_write_cache follows encoder-style RoPE positioning.
        # For pure decode we pass a local non-zero seq_lens_encoder view so RoPE indices align with seq_lens_decoder.
        fake_seq_lens_encoder = paddle.full_like(forward_meta.seq_lens_encoder, 1)
        max_dec_len_this_time = int(max_len_tensor_cpu[2].item())
        (
            attn_cu_seqlens_k,
            pre_cache_batch_ids,
            pre_cache_tile_ids_per_batch,
            pre_cache_num_blocks_cpu,
            kv_token_num_cpu,
        ) = pre_cache_len_concat(
            fake_seq_lens_encoder,
            forward_meta.seq_lens_decoder,
            forward_meta.seq_lens_this_time,
            max_dec_len_this_time,
            self.block_size,
        )
        kv_token_num = int(kv_token_num_cpu[0].item())

        q_decode, k_decode, v_decode, _ = gqa_rope_write_cache(
            qkv,
            cache_k,
            cache_v,
            forward_meta.cu_seqlens_q,
            attn_cu_seqlens_k,
            forward_meta.rotary_embs,
            forward_meta.seq_lens_this_time,
            fake_seq_lens_encoder,
            forward_meta.seq_lens_decoder,
            forward_meta.batch_id_per_token,
            forward_meta.block_tables,
            forward_meta.kv_batch_ids,
            forward_meta.kv_tile_ids_per_batch,
            forward_meta.kv_num_blocks_x_cpu,
            pre_cache_batch_ids,
            pre_cache_tile_ids_per_batch,
            pre_cache_num_blocks_cpu,
            getattr(layer, "q_norm_weight", None),
            getattr(layer, "k_norm_weight", None),
            cache_k_quant_scales,
            cache_v_quant_scales,
            getattr(layer, "cache_k_out_scale", None),
            getattr(layer, "cache_v_out_scale", None),
            getattr(layer, "cache_k_zp", None),
            getattr(layer, "cache_v_zp", None),
            metadata.kv_signal_data_list[layer.layer_id],
            kv_token_num=kv_token_num,
            max_seq_len=self.max_seq_len,
            rms_norm_eps=getattr(layer, "rms_norm_eps", 1e-6),
            use_neox_rotary_style=layer.use_neox_rotary_style,
            cache_quant_type=cache_quant_type_str,
            rope_3d=self.rope_3d,
        )

        token_num = int(q_decode.shape[0])
        if token_num == 0:
            return paddle.empty([0, self.num_heads * self.head_dim], dtype=qkv.dtype)
        if int(forward_meta.seq_lens_this_time.shape[0]) != 1:
            raise NotImplementedError("decode flash attention fallback currently requires effective batch size = 1.")

        # flash_attention expects [batch, seq_len, num_heads, head_dim].
        query_states = q_decode.unsqueeze(0)
        key_states = k_decode.unsqueeze(0)
        value_states = v_decode.unsqueeze(0)

        # Decode path does not need causal mask here.
        attn_output = F.flashmask_attention(
            query_states,
            key_states,
            value_states,
            dropout=0.0,
            causal=False,
        )
        if isinstance(attn_output, (tuple, list)):
            attn_output = attn_output[0]
        return attn_output[0].reshape([token_num, self.num_heads * self.head_dim])

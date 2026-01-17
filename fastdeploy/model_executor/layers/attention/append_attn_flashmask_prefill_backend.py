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
from typing import TYPE_CHECKING

import numpy as np
import paddle

from fastdeploy.model_executor.layers.attention.append_attn_backend import AppendAttentionBackend
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.attention.ops import (
    get_block_shape_and_split_kv_block,
    gqa_rope_write_cache,
    init_signal_layerwise,
    pre_cache_len_concat,
)
from fastdeploy.platforms import current_platform

if TYPE_CHECKING:
    from fastdeploy.model_executor.forward_meta import ForwardMeta


class AppendAttentionFlashMaskPrefillBackend(AppendAttentionBackend):
    """
    Experimental attention backend.

    - Prefill-only (first prefill) path: uses Paddle `flashmask_attention` v3, but still
      writes FastDeploy paged KV cache via `gqa_rope_write_cache` for subsequent decode.
    - Decode path: falls back to the standard AppendAttentionBackend implementation.

    Env knobs:
    - FD_USE_PADDLE_FLASHMASK_PREFILL=1: enable flashmask prefill path (when eligible)
    - FD_STRICT_PURE_PREFILL_DECODE=1: assert effective batch size==1 and pure prefill/decode only
    """

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

            seq_lens_this_time_list = (
                forward_meta.seq_lens_this_time.reshape([-1]).cpu().numpy().astype("int32").tolist()
            )
            effective_batch_size = sum(1 for x in seq_lens_this_time_list if int(x) > 0)
            if effective_batch_size != 1:
                raise AssertionError(
                    f"FD_STRICT_PURE_PREFILL_DECODE=1 requires effective batch size=1, got {effective_batch_size}."
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

            if (
                fa_version == 3
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
            ):
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

                seq_lens_this_time_list = (
                    forward_meta.seq_lens_this_time.reshape([-1]).cpu().numpy().astype("int32").tolist()
                )
                seq_lens_encoder_list = (
                    forward_meta.seq_lens_encoder.reshape([-1]).cpu().numpy().astype("int32").tolist()
                )
                active_bids = [
                    bid
                    for bid, (l_this, l_enc) in enumerate(zip(seq_lens_this_time_list, seq_lens_encoder_list))
                    if int(l_this) > 0 and int(l_enc) > 0
                ]
                bsz_active = len(active_bids)
                if bsz_active == 0:
                    return paddle.empty([0, self.num_heads * self.head_dim], dtype=qkv.dtype)

                q_dense = paddle.zeros([bsz_active, max_len_this_time, self.num_heads, self.head_dim], dtype=qkv.dtype)
                k_dense = paddle.zeros([bsz_active, max_len_this_time, self.kv_num_heads, self.head_dim], dtype=qkv.dtype)
                v_dense = paddle.zeros([bsz_active, max_len_this_time, self.kv_num_heads, self.head_dim], dtype=qkv.dtype)

                startend = np.full((bsz_active, 1, max_len_this_time, 1), max_len_this_time, dtype=np.int32)

                offset = 0
                row = 0
                for bid in range(len(seq_lens_this_time_list)):
                    l_this = int(seq_lens_this_time_list[bid])
                    if l_this <= 0:
                        continue
                    if int(seq_lens_encoder_list[bid]) <= 0:
                        raise NotImplementedError("flashmask prefill path does not support mixed prefill/decode batches.")
                    q_dense[row, :l_this] = q_packed[offset : offset + l_this]
                    k_dense[row, :l_this] = k_packed[offset : offset + l_this]
                    v_dense[row, :l_this] = v_packed[offset : offset + l_this]
                    startend[row, 0, l_this:, 0] = 0
                    offset += l_this
                    row += 1

                if offset != token_num:
                    raise RuntimeError("flashmask prefill pack/unpack mismatch.")

                if self.kv_num_heads != self.num_heads:
                    k_dense = paddle.repeat_interleave(k_dense, self.group_size, axis=2)
                    v_dense = paddle.repeat_interleave(v_dense, self.group_size, axis=2)

                import paddle.nn.functional as F

                startend_tensor = paddle.to_tensor(startend, place=q_dense.place)
                out_dense = F.flashmask_attention(
                    q_dense,
                    k_dense,
                    v_dense,
                    startend_tensor,
                    dropout=0.0,
                    causal=self.causal,
                    training=True,
                )

                out_packed = paddle.empty([token_num, self.num_heads * self.head_dim], dtype=out_dense.dtype)
                offset = 0
                row = 0
                for bid in range(len(seq_lens_this_time_list)):
                    l_this = int(seq_lens_this_time_list[bid])
                    if l_this <= 0:
                        continue
                    out_packed[offset : offset + l_this] = out_dense[row, :l_this].reshape(
                        [l_this, self.num_heads * self.head_dim]
                    )
                    offset += l_this
                    row += 1
                return out_packed

        return super().forward_mixed(q, k, v, qkv, compressed_kv, k_pe, layer, forward_meta)


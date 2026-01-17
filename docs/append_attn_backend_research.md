# AppendAttentionBackend 调研总结（FastDeploy）

本文聚焦 `fastdeploy/model_executor/layers/attention/append_attn_backend.py`，说明该 backend 的核心思路、prefill/extend 与 decode 在该 backend 中的实现方式、涉及的数据结构，并从“API 视角”描述 `forward_mixed` 的功能边界与输入/输出语义。

---

## 1. 这个 backend 的定位与核心思路

**一句话**：AppendAttentionBackend 用一个“写 KV Cache + 计算 Attention”的自定义 CUDA 融合算子，把 **prefill/extend** 与 **decode** 统一到同一个 `forward_mixed` 路径里处理，并通过预先计算好的 tile/block 映射表（`*_batch_ids`, `*_tile_ids_per_batch`, `*_num_blocks_*`）来高效 launch kernel。

核心设计点：

1. **“Append” 的含义**  
   每一步 forward 针对本次新增 token（`seq_lens_this_time` / `seq_lens_encoder`）：
   - 先把这些 token 的 **K/V 写入 KV cache**（append write cache）
   - 再用写好的 KV（包含历史 + 本次新增）计算本次 token 的 attention 输出

2. **统一 Mixed 模式：prefill/extend 与 decode 共用一个入口**  
   GPU 路径下 `ForwardMeta.forward_mode` 默认是 `MIXED`，因此 `AttentionBackend.forward()` 会一直走 `forward_mixed()`。  
   该 backend 并不实现 `forward_extend/forward_decode`，而是通过 `seq_lens_encoder/seq_lens_decoder` 的取值让 CUDA op “自动只跑需要的分支”。

3. **一次算“整个 batch”的 launch 规划，并复用到所有层**  
   `forward_mixed()` 在 `layer.layer_id == 0` 时调用 `get_block_shape_and_split_kv_block(...)`，把本 step 的 launch 规划（tile 拆分、block 数等）写进 `ForwardMeta` 的缓存张量中；后续层直接复用这些张量，避免每层重复计算。

4. **以 block-based KV cache + block table 组织长上下文**  
   KV cache 按 block（`block_size`）存储，`block_tables` 负责把每条序列的逻辑块映射到物理块，从而支持动态 batch、变长序列与可回收的 cache block 管理。

---

## 2. prefill/extend 与 decode 在该 backend 中如何实现

### 2.1 模式判定：靠 seq_lens 三元组“分流”

该 backend 主要依赖三组长度张量（均按 batch 维度组织）：

- `seq_lens_this_time`：本 step 每条序列要处理的 token 数（decode 通常为 1；chunked prefill/extend 可能 >1；spec decode 也可能 >1）。
- `seq_lens_encoder`：用于标识“本 step 是否存在 prefill/extend token”。通常 `>0` 表示该序列走 prefill/extend 分支。
- `seq_lens_decoder`：该序列在本 step 开始前的历史长度（已有 KV cache 的长度，用于决定 attention 的 KV 长度）。

在 `custom_ops/gpu_ops/append_attn/get_block_shape_and_split_kv_block.cu` 中：

- **decoder tile 拆分**会显式跳过 `seq_lens_encoder > 0` 的序列（`split_q_block(..., seq_lens_encoder, ...)` 内把这类序列的 `seq_len` 置 0）。
- **encoder tile 拆分**使用 `seq_lens_encoder` 作为 Q 的长度来源（`split_q_block(seq_lens_encoder, nullptr, ...)`）。
- **KV 写 cache 的 tile 拆分**使用 `seq_lens_encoder`（并结合 `seq_lens_decoder % block_size` 处理 block 对齐）。

因此，从“逻辑语义”上：

- **prefill/extend**：`seq_lens_encoder > 0` 的序列
- **decode**：`seq_lens_encoder == 0` 且 `seq_lens_decoder > 0` 的序列（代码中常称为 “just decode”）

### 2.2 prefill/extend 的执行路径（encoder 分支）

在 `custom_ops/gpu_ops/append_attention.cu` 的 `AppendAttentionKernel(...)` 中（简化描述）：

当 `max_enc_len_this_time > 0`（batch 中存在 `seq_lens_encoder > 0` 的序列）时：

1. **EncoderWriteCacheWithRopeKernel**  
   - 对本 step 的 Q/K（必要时也包含 V）做 RoPE（含 neox 风格与可选 3D RoPE）
   - 可选融合 qkv bias / scale、QK norm（`q_norm_weight/k_norm_weight`）、以及 KV cache 量化相关处理
   - 把 K/V **append 写入** `key_cache/value_cache`

2. **CascadeAppendAttentionKernel（encoder 形态）**  
   - 用 `encoder_batch_ids/encoder_tile_ids_per_batch/encoder_num_blocks_x_cpu` 指定 launch 的 grid.x 与 (batch, tile) 映射
   - 计算 prefill/extend token 的 attention 输出（查询长度来自 `seq_lens_encoder`，KV 长度结合历史 `seq_lens_decoder`）
   - 支持分区/分块（`max_partition_size/encoder_max_partition_size`）、causal、sliding window 等

### 2.3 decode 的执行路径（decoder 分支）

当 `max_just_dec_len_this_time > 0`（batch 中存在“纯 decode 序列”）时：

1. **DecoderWriteCacheWithRoPEKernel / SpeculateWriteCacheWithRoPEKernel**  
   - 把 decode token（或 speculative decode 的多 token）做 RoPE/（可选）QK norm 等预处理
   - 写入 `key_cache/value_cache`

2. **CascadeAppendAttentionKernel（decoder 形态）**  
   - 用 `decoder_batch_ids/decoder_tile_ids_per_batch/decoder_num_blocks_cpu` 做 decode 专用的 tile 组织与 kernel launch
   - KV 长度通常约等于 `seq_lens_decoder + seq_lens_this_time`

### 2.4 Mixed（prefill + decode 混合 batch）的并行策略

当同一个 step 同时存在 encoder 与 “just decode” 时：

- `append_attention.cu` 会创建一个额外的 `decoder_stream`，并用 CUDA event 做同步。
- 由于 event 在主流中被记录在 encoder kernel 入队之前，decode 分支可以与 encoder 分支在不同 stream **并行执行**（两者写 cache / 读写输出在 token 维度上应是相互独立的片段）。

---

## 3. 该 backend 涉及的关键数据结构

### 3.1 Python 侧：AppendAttentionMetadata

文件：`fastdeploy/model_executor/layers/attention/append_attn_backend.py`

`AppendAttentionMetadata` 继承自 `AttentionMetadata`，主要保存：

- `_dtype`：默认 dtype（来自 Paddle 默认 dtype）
- `_fuse_kernel_compute_dtype`：`bf16/fp16/fp32` 字符串，用于 qkv 为 `int32` 时选择计算精度
- `max_partition_size`：从环境变量 `FLAGS_max_partition_size` 读取（默认 1024），传给 CUDA kernel 做分区控制
- `encoder_max_partition_size`：通常设为 `max_model_len`
- `kv_signal_metadata / kv_signal_data_list`：PD disaggregation 场景下用于跨进程/跨阶段的 KV cache 同步信号

### 3.2 ForwardMeta：attention 相关字段（GPU 路径）

文件：`fastdeploy/model_executor/forward_meta.py`

AppendAttentionBackend 重点使用：

- 长度/布局
  - `seq_lens_encoder`, `seq_lens_decoder`, `seq_lens_this_time`
  - `batch_id_per_token`：padding removal 后 token -> batch 的映射
  - `cu_seqlens_q`：packed token 的累积 offset（类似 ragged batch 的 CSR 指针）
  - `block_tables`：每条序列的 block 映射表（逻辑 block -> 物理 block id）
- attention 辅助输入
  - `rotary_embs`：RoPE embedding（2D/3D；neox 风格对 shape 有额外约束）
  - `attn_mask`, `attn_mask_offsets`
- launch 规划缓存（由 `get_block_shape_and_split_kv_block` 填充）
  - `decoder_batch_ids`, `decoder_tile_ids_per_batch`, `decoder_num_blocks_cpu/device`
  - `encoder_batch_ids`, `encoder_tile_ids_per_batch`, `encoder_num_blocks_x_cpu`
  - `kv_batch_ids`, `kv_tile_ids_per_batch`, `kv_num_blocks_x_cpu`
  - `decoder_chunk_size_device`（主要用于 MLA 特殊路径）
  - `max_len_tensor_cpu`：长度摘要数组（CUDA 侧读前 6 个 int）

### 3.3 launch 规划缓存的初始化（share_inputs）

文件：`fastdeploy/worker/gpu_model_runner.py`

`GPUModelRunner._initialize_attn_backend()` 调用 `allocate_launch_related_buffer(...)` 生成并放入 `share_inputs`：

- decoder/encoder/kv 的 `batch_ids` 与 `tile_ids_per_batch`（int32）
- `*_num_blocks_*`（shape `[1]` 的 int32；有的在 CPU/pin_memory）
- `max_len_tensor_cpu`（CPU int32，shape `[9]`）

这些 buffer 的容量按 `max_batch_size * max_model_len` 与 block/tile 配置（`encoder_block_shape_q=64`, `decoder_block_shape_q=16`）估算，保证最坏情况也能覆盖。

### 3.4 KV cache 的形状与索引方式

文件：`fastdeploy/model_executor/layers/attention/append_attn_backend.py`

`get_kv_cache_shape(max_num_blocks, kv_cache_quant_type)`：

- `key_cache`：`[max_num_blocks, kv_num_heads, block_size, head_dim]`
  - 若 `kv_cache_quant_type == "int4_zp"`：最后一维为 `head_dim // 2`
- `value_cache`：与 `key_cache` 同 shape

并且在 `forward_mixed()` 中按 layer id 从 `forward_meta.caches` 取出对应层的 cache：

- 常规：`caches[2 * layer_id]` / `caches[2 * layer_id + 1]`（K/V）
- `block_wise_fp8`：`caches[4 * layer_id + 0..3]`（K/V + K/V scales）

---

## 4. `forward_mixed` 会完成的 API 功能（语义描述）

文件：`fastdeploy/model_executor/layers/attention/append_attn_backend.py`

从调用者视角，`AppendAttentionBackend.forward_mixed(...)` 的职责可以概括为：

1. **准备与复用本 step 的 attention launch 规划**  
   在 `layer.layer_id == 0` 时，根据 `seq_lens_encoder/decoder/this_time` 调用 `get_block_shape_and_split_kv_block(...)` 填充 `ForwardMeta` 中的 `*_batch_ids/*_tile_ids/*_num_blocks/max_len_tensor_cpu`，使后续所有 attention layer 都能复用同一份规划。

2. **按 layer id 选择本层 KV cache（以及量化/scale 参数）**  
   支持普通 cache、int4/int8/fp8 量化 cache、block-wise fp8（scales 存于 `caches` 列表）。

3. **调用融合 CUDA op：写 KV cache + 计算 attention 输出**  
   - encoder 分支：处理 `seq_lens_encoder > 0` 的序列（prefill/extend）
   - decoder 分支：处理 “just decode” 序列
   - mixed 时可能并行执行

4. **输出张量语义**  
   返回本层 attention 的输出，token 维度与 `qkv` 一致（即 padding removal 后的 packed token 顺序），形状为：
   - `[..., num_heads * head_dim]`（实现中构造为 `[token_nums, q_num_heads * head_dims]`）
   - dtype 取决于 `qkv.dtype`、`compute_type`、以及是否启用输出量化（`out_scale/quant_max_bound`）

补充说明：

- 该函数签名包含 `q/k/v/compressed_kv/k_pe`，但在该 backend 中实际主要使用 `qkv`（其余参数主要用于统一 attention backend 接口，或给其他 backend 使用）。
- RoPE 的 shape 约束会在 Python 侧做断言检查，以避免进入 CUDA op 后出现难定位的形状错误。

---

## 5. 关键文件索引（便于继续深入）

- Python backend：`fastdeploy/model_executor/layers/attention/append_attn_backend.py`
- ForwardMeta 定义：`fastdeploy/model_executor/forward_meta.py`
- launch 规划缓存分配：`fastdeploy/worker/gpu_model_runner.py`
- CUDA：计算 block/tile 映射与长度摘要：`custom_ops/gpu_ops/append_attn/get_block_shape_and_split_kv_block.cu`
- CUDA：融合 op（写 cache + attention）：`custom_ops/gpu_ops/append_attention.cu`
- CUDA：CascadeAppendAttentionKernel 分发：`custom_ops/gpu_ops/append_attn/append_attention_kernel.h`

---

## 6. 实验性改造：prefill 用 Paddle FlashMask(v3)，decode 仍用 paged KV cache

仓库内提供了一个实验性 backend：`fastdeploy/model_executor/layers/attention/append_attn_flashmask_prefill_backend.py`，在满足非常基础前提时，prefill 阶段改用 Paddle 的 `paddle.nn.functional.flashmask_attention`（要求走 v3），但仍通过 FastDeploy 的 `gqa_rope_write_cache` 把 KV 写入 `forward_meta.caches`，从而 decode 阶段继续走原 `append_attention` 在 paged KV cache 上计算。

- 选择 backend：设置 `FD_ATTENTION_BACKEND=APPEND_ATTN_FLASHMASK_PREFILL`
- prefill 路径开关：设置环境变量 `FD_USE_PADDLE_FLASHMASK_PREFILL=1`
- 约束（不满足会回退到原 `append_attention` 路径）：
  - 仅支持 prefill-only step（该 step 不允许有 decode token）
  - 仅支持“首次 prefill”（不支持 chunked prefill / 已存在历史 KV）
  - 不支持 KV cache 量化（要求 `cache_quant_type_str == "none"`）
  - 不使用 `attn_mask/attn_mask_offsets`，不支持 `sliding_window`
  - 要求 `FLAGS_flash_attn_version=3`，且在 `FLAGS_cudnn_deterministic=1 && head_dim>128` 时禁用（否则 Paddle 会强制走 v2）
  - 要求 `qkv.dtype == bfloat16`（当前 `gqa_rope_write_cache` CUDA 实现的限制）

- 调试用强约束（会直接 `assert`/抛错，不会回退）：
  - 设置 `FD_STRICT_PURE_PREFILL_DECODE=1`：要求有效 batch size=1，且每个 step 只能是“首次 prefill”或“纯 decode”，并强制 chunked prefill 关闭。

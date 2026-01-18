#!/bin/bash

# 测试实验性 AppendAttentionBackend 的脚本
# 设置 conda 环境 lsy_fd_env

echo "=== 测试实验性 AppendAttentionBackend ==="

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsy_fd_env

echo "当前Python环境: $(which python)"
echo "Python版本: $(python --version)"

# Paddle Trainer相关
unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT
unset PADDLE_TRAINERS_NUM
export PADDLE_TRAINERS_NUM=1

# 设置关键环境变量
export FD_ATTENTION_BACKEND="APPEND_ATTN_FLASHMASK_PREFILL"
export FD_USE_PADDLE_FLASHMASK_PREFILL=1
export FD_STRICT_PURE_PREFILL_DECODE=1
export FD_DISABLE_CHUNKED_PREFILL=1
export FLAGS_flash_attn_version=3
export FLAGS_cudnn_deterministic=0
# 限制并发数为1，设置批处理参数
export FD_MAX_NUM_SEQS=1
export FD_MAX_MODEL_LEN=2048
export FD_MAX_NUM_BATCHED_TOKENS=2048

# 打印环境变量设置
echo "环境变量设置:"
echo "FD_ATTENTION_BACKEND=$FD_ATTENTION_BACKEND"
echo "FD_USE_PADDLE_FLASHMASK_PREFILL=$FD_USE_PADDLE_FLASHMASK_PREFILL"
echo "FD_STRICT_PURE_PREFILL_DECODE=$FD_STRICT_PURE_PREFILL_DECODE"
echo "FD_DISABLE_CHUNKED_PREFILL=$FD_DISABLE_CHUNKED_PREFILL"
echo "FLAGS_flash_attn_version=$FLAGS_flash_attn_version"
echo "FLAGS_cudnn_deterministic=$FLAGS_cudnn_deterministic"

# 运行 Python 测试脚本
echo "=== 运行 Python 测试脚本 ==="
python test_experimental_backend.py

# 可选：测试标准 backend 作为对比
echo "=== 测试标准 backend 作为对比 ==="
export FD_ATTENTION_BACKEND="APPEND_ATTN"
unset FD_USE_PADDLE_FLASHMASK_PREFILL
unset FD_STRICT_PURE_PREFILL_DECODE

echo "环境变量设置（标准模式）:"
echo "FD_ATTENTION_BACKEND=$FD_ATTENTION_BACKEND"

python test_experimental_backend.py

echo "=== 测试完成 ==="
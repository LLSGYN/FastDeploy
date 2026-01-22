#!/usr/bin/env python3
"""
测试实验性 AppendAttentionBackend 的 Python 脚本
"""

import os
import sys
import time
import inspect
import traceback
from fastdeploy import LLM, SamplingParams

def print_environment_info():
    """打印环境信息和配置"""
    print("=" * 60)
    print("环境配置信息:")
    import fastdeploy
    print(f"fastdeploy路径: {fastdeploy.__file__}")
    print(f"Python可执行文件: {sys.executable}")
    print(f"sys.flags.optimize: {sys.flags.optimize} (PYTHONOPTIMIZE={os.getenv('PYTHONOPTIMIZE', '未设置')})")
    print(f"FD_ATTENTION_BACKEND: {os.getenv('FD_ATTENTION_BACKEND', '未设置')}")
    print(f"FD_USE_PADDLE_FLASHMASK_PREFILL: {os.getenv('FD_USE_PADDLE_FLASHMASK_PREFILL', '未设置')}")
    print(f"FD_STRICT_PURE_PREFILL_DECODE: {os.getenv('FD_STRICT_PURE_PREFILL_DECODE', '未设置')}")
    print(f"FLAGS_flash_attn_version: {os.getenv('FLAGS_flash_attn_version', '未设置')}")
    print("RR Attention 测试参数: stride=8, threshold=0.90")
    print("=" * 60)

def _bool_env(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes", "y")


def test_rr_attention_op():
    """测试 paddlefleet rr_attention 算子是否可用"""
    print("\n=== 测试 paddlefleet rr_attention 算子（stride=8, threshold=0.90） ===")

    try:
        import paddle
    except Exception as e:
        print(f"⚠️ 无法导入 paddle，跳过 rr_attention 测试: {e}")
        return "SKIP"

    try:
        import paddlefleet  # noqa: F401
    except Exception as e:
        print(f"⚠️ paddlefleet 未安装或不可用，跳过 rr_attention 测试: {e}")
        return "SKIP"

    try:
        import paddlefleet

        if not hasattr(getattr(paddlefleet, "ops", None), "rr_attention"):
            print("⚠️ paddlefleet.ops.rr_attention 不存在，跳过 rr_attention 测试")
            return "SKIP"

        if not paddle.is_compiled_with_cuda():
            print("⚠️ Paddle 未编译 CUDA，跳过 rr_attention 测试")
            return "SKIP"

        try:
            paddle.device.set_device("gpu:0")
        except Exception as e:
            print(f"⚠️ 无可用 GPU，跳过 rr_attention 测试: {e}")
            return "SKIP"

        # 采用与 FastDeploy flashmask/rr_attention 路径一致的输入布局
        bs = 1
        seqlen = 16
        num_heads = 2
        head_dim = 64

        q = paddle.randn([bs, seqlen, num_heads, head_dim], dtype="float32").astype("bfloat16")
        k = paddle.randn([bs, seqlen, num_heads, head_dim], dtype="float32").astype("bfloat16")
        v = paddle.randn([bs, seqlen, num_heads, head_dim], dtype="float32").astype("bfloat16")
        startend_tensor = paddle.full([bs, 1, seqlen, 1], seqlen, dtype="int32")

        out = paddlefleet.ops.rr_attention(
            q,
            k,
            v,
            startend_tensor,
            dropout=0.0,
            causal=True,
            training=True,
            threshold=0.90,
            stride=8,
        )

        assert list(out.shape) == [bs, seqlen, num_heads, head_dim], f"unexpected out.shape={list(out.shape)}"
        print("✅ rr_attention 算子运行成功")
        return "PASS"
    except Exception as e:
        print(f"❌ rr_attention 算子运行失败: {e}")
        traceback.print_exc()
        return "FAIL"


def test_basic_inference():
    """基本推理测试"""
    print("\n=== 开始基本推理测试 ===")
    
    try:
        # 使用一个小模型进行测试
        # ERNIE-4.5-0.3B-Paddle 是一个适合测试的小模型
        model_name = "baidu/ERNIE-4.5-0.3B-Paddle"
        
        print(f"正在加载模型: {model_name}")
        
        # 创建采样参数
        sampling_params = SamplingParams(
            top_p=0.95,
            max_tokens=100,
            temperature=0.7
        )
        
        # 加载模型 - 使用最小的配置
        start_time = time.time()
        graph_optimization_config = None
        if os.getenv("FD_TEST_DISABLE_CUDAGRAPH", "0").lower() in ("1", "true"):
            graph_optimization_config = {"use_cudagraph": False, "graph_opt_level": 0}
            print(f"⚠️ 已禁用CUDAGraph用于调试: {graph_optimization_config}")
        llm = LLM(
            model=model_name,
            tensor_parallel_size=1,  # 单卡
            max_model_len=2048,      # 减小模型长度以减少内存需求
            gpu_memory_utilization=0.7,  # 进一步增加内存利用率
            max_num_seqs=1,         # 限制并发数为1
            max_num_batched_tokens=2048,  # 设置与max_model_len相同的值
            graph_optimization_config=graph_optimization_config,
        )
        load_time = time.time() - start_time
        print(f"模型加载完成，耗时: {load_time:.2f}秒")
        
        # 准备测试消息
        test_messages = [
            [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "请用一句话介绍你自己"}
            ]
        ]
        
        print("开始推理...")
        inference_start = time.time()
        
        # 进行推理 - 设置use_tqdm=True来绕过FastDeploy的bug
        outputs = llm.chat(test_messages, sampling_params, use_tqdm=True)
        
        inference_time = time.time() - inference_start
        
        # 输出结果
        print(f"推理完成，耗时: {inference_time:.2f}秒")
        print("\n推理结果:")
        for i, output in enumerate(outputs):
            print(f"请求 {i+1}:")
            print(f"  输入: {output.prompt}")
            print(f"  输出: {output.outputs.text}")
            print(f"  生成的token数: {len(output.outputs.token_ids)}")
        
        return True
        
    except Exception as e:
        print(f"推理测试失败: {e}")
        traceback.print_exc()
        return False

def test_experimental_features():
    """测试实验性功能"""
    print("\n=== 测试实验性功能 ===")
    
    try:
        # 检查实验性 backend 是否正确加载
        from fastdeploy.model_executor.layers.attention.append_attn_flashmask_prefill_backend import AppendAttentionFlashMaskPrefillBackend
        
        print("实验性 backend 类已成功导入")
        print(f"backend源码路径: {inspect.getfile(AppendAttentionFlashMaskPrefillBackend)}")
        
        # 检查环境变量设置
        use_flashmask_prefill = os.getenv("FD_USE_PADDLE_FLASHMASK_PREFILL", "0")
        strict_mode = os.getenv("FD_STRICT_PURE_PREFILL_DECODE", "0")
        
        print(f"FD_USE_PADDLE_FLASHMASK_PREFILL: {use_flashmask_prefill}")
        print(f"FD_STRICT_PURE_PREFILL_DECODE: {strict_mode}")
        
        if use_flashmask_prefill == "1":
            print("✅ Paddle FlashMask Prefill 模式已启用")
        else:
            print("⚠️ Paddle FlashMask Prefill 模式未启用")
            
        if strict_mode == "1":
            print("✅ 严格模式已启用（仅支持纯 prefill/decode）")
        else:
            print("⚠️ 严格模式未启用")
            
        return True
        
    except ImportError as e:
        print(f"无法导入实验性 backend: {e}")
        return False
    except Exception as e:
        print(f"实验性功能测试失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("实验性 AppendAttentionBackend 测试脚本")
    print("=" * 60)
    
    # 打印环境信息
    print_environment_info()
    
    # 测试实验性功能
    experimental_ok = test_experimental_features()

    # rr_attention 算子测试（可选：无 paddlefleet / 无 GPU 时自动跳过）
    rr_attn_status = test_rr_attention_op()
    
    # 进行推理测试
    inference_ok = test_basic_inference()
    
    # 总结结果
    print("\n" + "=" * 60)
    print("测试结果总结:")
    print(f"实验性功能检查: {'✅ 通过' if experimental_ok else '❌ 失败'}")
    if rr_attn_status == "PASS":
        rr_attn_summary = "✅ 通过"
    elif rr_attn_status == "SKIP":
        rr_attn_summary = "⚠️ 跳过"
    else:
        rr_attn_summary = "❌ 失败"
    print(f"rr_attention 算子测试: {rr_attn_summary}")
    print(f"推理测试: {'✅ 通过' if inference_ok else '❌ 失败'}")
    
    if experimental_ok and inference_ok and rr_attn_status != "FAIL":
        print("\n🎉 测试通过！实验性 backend 可以正常运行")
    else:
        print("\n⚠️ 测试出现问题，请检查环境配置和错误信息")
    
    print("=" * 60)

if __name__ == "__main__":
    main()

# FastDeploy 与 HuggingFace `generate` 无法严格对齐的原因（重点：Greedy）

## 1. 结论先行

在当前实现下，FastDeploy（FD）与 HuggingFace Transformers（HF）即使参数尽量对齐，也通常只能做到“统计上接近”或“多数样本一致”，难以保证**严格对齐**：

1. 每步 token 完全一致（token-level strict equality）
2. 全链路数值逐位一致（bitwise equality）

其根因不是单一参数，而是两套生成系统在**选择算子、logits 处理、停止机制、调度与数值路径**上的架构性差异。

---

## 2. 严格对齐的数学条件

令第 \(t\) 步模型原始 logits 为 \(z_t \in \mathbb{R}^{|V|}\)，历史 token 为 \(y_{<t}\)。

若要严格对齐，必须同时满足：

\[
\forall t,\quad y_t^{FD} = y_t^{HF}
\]

更强地（bitwise）：

\[
\forall t,\quad \tilde z_t^{FD} = \tilde z_t^{HF}
\]

其中 \(\tilde z_t\) 是经过全部 logits processor / penalty / mask 后的最终打分。  
任一步不满足，后续自回归状态会分叉，误差会累积放大。

---

## 3. HF 的 Greedy 路径（源码行为）

HF 在 `GenerationMode.GREEDY_SEARCH` 与 `SAMPLE` 共用 `_sample`，但当 `do_sample=False` 时，token 选择是直接 `argmax`：

- `transformers/src/transformers/generation/utils.py:2205`
- `transformers/src/transformers/generation/utils.py:3246`
- `transformers/src/transformers/generation/utils.py:3251`

即：

\[
y_t^{HF}=\arg\max_i \tilde z_{t,i}^{HF}
\]

另外，HF 的采样 warper（temperature/top-k/top-p）仅在 `do_sample=True` 才生效：

- `transformers/src/transformers/generation/utils.py:1019`

`min_new_tokens` 的实现是“在达到下限前把 EOS 置为 \(-\infty\)”：

- `transformers/src/transformers/generation/utils.py:941`
- `transformers/src/transformers/generation/utils.py:946`
- `transformers/src/transformers/generation/logits_process.py:232`
- `transformers/src/transformers/generation/logits_process.py:237`

可写为：

\[
t < m \Rightarrow \tilde z_{t,\text{eos}}=-\infty
\]

---

## 4. FD 的主路径（与 HF 不同）

FD 常规 sampler 路径（非 speculative 的 normal sampler）是：

- `fastdeploy/model_executor/layers/sample/sampler.py:486`
- `fastdeploy/model_executor/layers/sample/sampler.py:526`
- `fastdeploy/model_executor/layers/sample/sampler.py:529`

即先做：

\[
p_t = \operatorname{softmax}(\tilde z_t^{FD})
\]

默认会走 `top_k_top_p_sampling` 族算子选 token。

在默认 `FD_SAMPLING_CLASS=base` 下：

- `fastdeploy/envs.py:54`
- `fastdeploy/model_executor/layers/sample/ops/top_k_top_p_sampling.py:63`
- `fastdeploy/model_executor/layers/sample/ops/top_k_top_p_sampling.py:88`

该分支本质是 top-p 采样内核；`top_k` 在该分支不主导行为。

补充：当前仓库已在 `Sampler.forward_cuda` 中加入 `do_sample=False` 的显式 `argmax` 分支（接近 HF greedy）：

- `fastdeploy/model_executor/layers/sample/sampler.py:526`
- `fastdeploy/model_executor/layers/sample/sampler.py:551`

但这仍不意味着两端自动严格等价（后续章节解释）。

---

## 5. 为什么“参数对齐了”也仍可能不严格一致

## 5.1 选择算子差异：`argmax` 路径与采样路径并存

HF greedy：

\[
y_t^{HF}=\arg\max_i \tilde z_{t,i}^{HF}
\]

在没有显式 `do_sample=False` 标志时，FD 常规仍是采样核；此时即使调成 `rejection + top_k=1`，也只是让采样分布退化到单候选，接近 greedy，不等于共享同一实现路径。

- `custom_ops/gpu_ops/sample_kernels/rejection_top_p_sampling.cu:19`
- `custom_ops/gpu_ops/sample_kernels/rejection_top_p_sampling.cu:49`

## 5.2 请求参数会被 FD 预处理改写

FD 预处理里：

- `fastdeploy/input/text_processor.py:266`
- `fastdeploy/input/text_processor.py:268`
- `fastdeploy/input/text_processor.py:269`

当 \(T < \epsilon\) 时会被改为 \(T=1\)，当 `top_p < eps` 会被抬到 `eps`。  
这与 HF “`do_sample=False` 直接 argmax，不依赖温度采样”的语义不同。

## 5.3 `min_new_tokens` / `min_tokens` 语义不等价

HF：在选择前屏蔽 EOS（见第 3 节）。  
FD：在 stop kernel 里通过 `current_step >= min_tokens` 决定“能否停”，并非先屏蔽 EOS：

- `custom_ops/gpu_ops/stop_generation_multi_ends.cu:49`
- `custom_ops/gpu_ops/stop_generation_multi_ends.cu:69`
- `custom_ops/gpu_ops/stop_generation_multi_ends.cu:77`

这会导致同样的 `min_*` 参数下，两边在“达到最小长度前”对 EOS 的处理路径不同。

## 5.4 logits processor 链路和顺序不同

HF 由 `LogitsProcessorList` 逐个处理（且 do_sample 控制 warper 是否加入）：

- `transformers/src/transformers/generation/utils.py:843`
- `transformers/src/transformers/generation/utils.py:1019`

FD 使用 fused/custom op 统一做 repetition/frequency/presence/temperature/bad-words/min-len-eos 等：

- `fastdeploy/model_executor/layers/sample/sampler.py:505`
- `custom_ops/gpu_ops/token_penalty_multi_scores.cu:107`
- `custom_ops/gpu_ops/token_penalty_multi_scores.cu:132`
- `custom_ops/gpu_ops/token_penalty_multi_scores.cu:137`

FD 内核中的核心变换可写为（单 token 维度）：

\[
\hat z_i=
\begin{cases}
z_i\cdot \alpha, & z_i<0\ \land\ i\in R \\
z_i/\alpha, & z_i\ge0\ \land\ i\in R \\
z_i, & i\notin R
\end{cases}
\]

\[
\tilde z_i = \frac{\hat z_i - \beta\,c_i - \gamma\,\mathbf{1}[c_i>0]}{T}
\]

其中 \(R\) 为“在 prompt 或历史中出现过”的集合，\(c_i\) 为重复计数。  
即便看起来“概念相同”，实现路径、执行顺序、内核细节不同，仍可能产生微差。

## 5.5 停止机制实现不同（EOS / stop sequences）

FD 通过 `set_stop_value_multi_ends` 在后处理阶段更新 `stop_flags/next_tokens`：

- `fastdeploy/model_executor/pre_and_post_process.py:360`
- `custom_ops/gpu_ops/stop_generation_multi_ends.cu:26`

HF 通过 `StoppingCriteria` 与 `unfinished_sequences` 在 `_sample` loop 内管理：

- `transformers/src/transformers/generation/utils.py:3254`
- `transformers/src/transformers/generation/utils.py:3262`

机制不同意味着“同一时刻是否结束、结束后写入什么 token”不一定一致。

## 5.6 数值路径差异会放大到 token 分叉

FD/HF 在 attention kernel、fused op、调度（如连续批处理、chunked prefill）路径上并不共享实现。  
即使参数一致，也会有 \(\delta_t\) 级别数值扰动。设 top-1 与 top-2 间隔：

\[
\Delta_t = \tilde z_{t,(1)} - \tilde z_{t,(2)}
\]

当扰动满足：

\[
(\delta_{t,(2)} - \delta_{t,(1)}) > \Delta_t
\]

就会发生 argmax 翻转，从该步开始序列分叉。

## 5.7 采样随机状态推进机制不同（补充）

在存在采样的路径上，FD worker 侧会推进 `infer_seed`：

- `fastdeploy/worker/gpu_model_runner.py:196`
- `fastdeploy/worker/gpu_model_runner.py:2058`

这与 HF 的随机数状态推进实现并不共享同一代码路径。  
因此即使给了“同一个 seed 数值”，也不意味着两边逐步采样轨迹可严格重合。

---

## 6. 关于“Greedy 对齐”的现实边界

可以做的工程手段（例如强制 `rejection + top_k=1`、显式 eos 对齐、关闭不必要特性）会显著提升一致率；  
但在当前两套系统的架构下，这些手段通常仍是“逼近同一行为”，而非“同一实现的严格等价”。

换句话说：

- 可以追求：高一致率、可复现趋势、误差可解释
- 难以保证：对所有输入的逐 token 严格一致与 bitwise 一致

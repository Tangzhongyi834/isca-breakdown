# LLM Runtime Breakdown Profiling

用于研究 LLaMA / Mistral 的 Linear 精度从 FP16 降至 W8A8、W4A4 时，prefill 计算时间构成如何随输入长度变化。原始实现要求保留在 [readme.md](readme.md)，此文件为运行说明。

**Linear latency means GEMM kernel latency only. Quantization/dequantization overhead is excluded.**

**W4A4 means true INT4 activation × INT4 weight GEMM. Weight-only INT4 is not accepted.**

不评估 perplexity、下游准确率或量化误差，不预设任何精度一定更快、非线性占比一定上升。程序不会生成预测值或补齐缺失的测量结果。

## 1. 环境安装

Linux、NVIDIA CUDA GPU，驱动须支持所安装 PyTorch 的 CUDA 版本。建议在独立环境中安装：

本机也可直接使用已有的 `million` 环境：

```bash
conda activate million
python profile_llm.py --help
```

如需新建环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

已验证环境：NVIDIA H20（SM90），包括本机 `million` 的 PyTorch 2.5.1/CUDA 12.1、Transformers 4.46.3、Datasets 2.18.0，以及 PyTorch 2.6.0/CUDA 12.4、Transformers 4.48.3。验收记录见 [VALIDATION.md](VALIDATION.md)。依赖限制在相应 API 范围内；升级 PyTorch/Transformers 后应重新执行 CUDA 测试。必须启用可采集 CUDA kernel 的 PyTorch profiler/CUPTI；捕获不到 kernel 时不会输出正式汇总。

需要访问 Hugging Face。LLaMA 等 gated checkpoints 需要先在 Hugging Face 获得访问权限并完成登录，或通过 `HF_TOKEN` 提供令牌。也可使用已下载的本地 checkpoint 路径；程序不打印认证信息。`--local-files-only` 限制模型及 tokenizer 的下载；数据集离线还需要本地 Datasets 缓存和相应离线配置。

## 2. 模型与精度

模型始终由 `AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.float16, attn_implementation="eager")` 加载，按 `model.config.model_type` 选择适配器，不重写 Transformer。

支持 `llama` / `mistral`，例如：

- `meta-llama/Llama-2-7b-hf`
- `meta-llama/Meta-Llama-3-8B`
- `mistralai/Mistral-7B-v0.1`

`--model` 必须显式指定。包括 LM head 在内的所有 `nn.Linear` 在替换前按模块类型记录身份，替换后仍保持原始名称。FP16 backend 将 bias 与 GEMM 分离；W8A8 backend 使用相同分离方式。

| 模式 | 实际计算 | 当前行为 |
| --- | --- | --- |
| FP16 | FP16 × FP16 CUDA GEMM | 支持 |
| W8A8 | INT8 × INT8 → INT32，`torch._int_mm` | 支持；先校验真实 CUDA kernel 和整数乘法结果 |
| W4A4 | 必须为 INT4 × INT4 | 当前没有集成合格 backend，立即明确报错 |

W8A8 使用静态 per-output-channel weight scale、动态 per-token activation scale。激活量化、缩放、padding/格式转换、反量化各自划分类别。权重量化在初始化阶段执行，不能混入每次 forward；记录其 setup-only 属性。输入维度不满足整数 GEMM 对齐约束时执行显式 padding，padding 归入转换开销，实际 GEMM 包括对齐后的计算。

## 3. 准备 WikiText-2

固定 `Salesforce/wikitext` / `wikitext-2-raw-v1` / `test`。按原始顺序过滤完全空白的条目后，以 `\n\n` 拼接；不修改其他文本内容。使用 checkpoint 自己的 fast tokenizer，`add_special_tokens=False`。

```bash
python profile_llm.py --model meta-llama/Llama-2-7b-hf --prepare-data-only
```

默认生成 32 个不重叠的 4096-token 窗口。窗口起点为 `token_offset + sample_id * 4096`，默认 `--token-offset 0`。512/1024/2048 均取该窗口前缀；每次输入严格 `(1, seq_len)`，不补 BOS、不 padding、不 shuffle。文本 token 不足时明确失败，不悄悄减少样本数。

缓存位置为 `data/wikitext2/<model_type>/<identity-hash>/samples.pt`。identity 包含完整模型名、revision、样本数和 offset，避免同一架构下不同 checkpoint 共用错误 tokenizer。缓存保存 token tensor、SHA256、数据集指纹、tokenizer 信息及预处理元数据。所有精度读取相同缓存；读取时核对元数据、形状和 checksum。

可通过 `--revision` 固定模型/tokenizer 的版本，通过 `--cache-dir` 指定缓存根目录。若需要更换 tokenizer 版本，应使用新的 revision 或缓存目录。

## 4. FP16 profiling

```bash
python profile_llm.py \
  --model meta-llama/Llama-2-7b-hf \
  --precision fp16 \
  --dataset Salesforce/wikitext --dataset-config wikitext-2-raw-v1 --split test \
  --seq-lens 512 1024 2048 4096 \
  --num-samples 32 --batch-size 1 --warmup 5 --repeats 3 \
  --phase prefill --device cuda:0
```

各长度独立 warmup，使用 sample 0。另有不计入测量的输出 sanity forward。正式运行均为 `eval()`、`torch.inference_mode()`、`use_cache=False`，强制 eager attention、禁用 TF32。检查输入、模型设备、dtype、logits 形状及 finite 值。只支持 prefill，decode 不能混入当前结果。

## 5. W8A8 profiling

```bash
python profile_llm.py \
  --model meta-llama/Llama-2-7b-hf --precision w8a8 \
  --seq-lens 512 1024 2048 4096 --num-samples 32 --phase prefill --device cuda:0

python profile_llm.py \
  --model mistralai/Mistral-7B-v0.1 --precision w8a8 \
  --seq-lens 512 1024 2048 4096 --num-samples 32 --phase prefill --device cuda:0
```

会输出 Weight/Activation Precision、GEMM backend、实际 kernel 名称，并保存算术校验与 kernel 证据。`torch._int_mm` 为内部 API，因此固定并验证 PyTorch 版本；不能运行或不能证明真实 INT8 kernel 时明确失败，不切换浮点计算。

## 6. W4A4 backend 要求

```bash
python profile_llm.py --model meta-llama/Llama-2-7b-hf --precision w4a4
```

当前实现将返回退出码 2：

```text
True W4A4 CUDA backend is unavailable.
```

`W4A4Backend` 是明确拒绝替代计算的接口位置，当前不是可执行 INT4 实现。后续需在其中接入支持目标 GPU 的真实 INT4×INT4 CUDA kernel，分别实现 scale/quantize/packing/GEMM/dequantize，并同步扩展严格 kernel 验证与测试。不能仅改变 backend 名称或用 W4A16、AWQ、GPTQ、bitsandbytes 代替。当前不提供 projected W4A4，因此不会产生 projected CSV。

## 7. 计时与分类口径

| 字段/类别 | 定义 |
| --- | --- |
| `linear_ms` | Projection、FFN、LM head 的 GEMM kernel 时长 |
| `softmax_ms` | attention 内 softmax |
| `rope_ms` | rotary embedding 生成及完整 `apply_rotary_pos_emb` |
| `silu_ms` | 仅 SiLU，不包含 gate × up |
| `rmsnorm_ms` | RMSNorm 内 square/reduce/rsqrt/scale 等全部 kernel |
| `attention_matmul_ms` | QKᵀ 和 PV 的 GEMM kernel，独立于 Linear |
| `quantization_overhead_ms` | quantize/dequantize/requantize/scale/packing/conversion |
| `other_compute_ms` | 其余 GPU 计算 kernel，例如 residual、mask、gate × up |

完整 RMSNorm/RoPE 内部的操作归入对应 nonlinear 类别。算子范围外的独立 dtype/layout 转换归入 conversion，因此 FP16 也可能有 conversion 开销。Attention MatMul 内部辅助拷贝从 GEMM 中分离到 conversion。所有 kernel 互斥计数，未知/无法关联的 kernel 触发失败，不能悄悄忽略。

`CUDAEventTimer` 记录开始/结束事件，forward 结束后统一同步，绝不逐算子 synchronize。CUDA Event 的算子时间跨度可能包含 CPU 提交空隙，不能直接当作纯 kernel latency：因此论文各类别使用 PyTorch profiler/CUPTI 实际 CUDA kernel duration，依赖 External ID 和 ATen 上下文关联；事件跨度同时保存为诊断证据。不会使用 CPU wall clock 测量 GPU 算子。

每个 repeat 先运行一次不加 instrumentation 的相同输入，CUDA Events 测量其 `physical_total_ms`；再运行一次带 instrumentation 的 forward 获取 kernel 分类。后者的总事件跨度单独存为 `profiled_physical_ms`。两次执行不要求逐项严格相等，不能把 profiler 开销算成 Other Compute。还会捕获 DMA memcpy/memset：GEMM 内的 workspace 初始化和拷贝单列 conversion，不能并入 Linear。`gpu_kernel_total_ms` 仅含 kernel，`gpu_activity_total_ms` 包含 kernel 与这些内存活动；二者均不含 GPU idle gap，因此不等于物理时间。

```text
nonlinear_ms = softmax_ms + rope_ms + silu_ms + rmsnorm_ms
other_ms = attention_matmul_ms + other_compute_ms
compute_total_ms = linear_ms + nonlinear_ms + other_ms
linear_pct = linear_ms / compute_total_ms * 100
nonlinear_pct = nonlinear_ms / compute_total_ms * 100
other_pct = other_ms / compute_total_ms * 100
```

量化和转换开销不进入这些分母。直接对原始 kernel 列表求和，独立检查分类是否遗漏或重复；超过 5% 的一致性误差会提示并拒绝发布该配置。

## 8. 重复、统计及公平性

默认每个 sample 测 3 次，逐基础类别取 median；再在 32 个 sample 上逐基础类别取 median。Nonlinear、Compute Total、比例等派生字段在每次聚合后重新计算，以保持可加性。**这些派生总和不声称是各 repeat 总时间的独立 median**；每项原始 sample 指标的 mean/median/std/min/max/p10/p90 另存 statistics CSV。std 使用样本标准差；单一样本为 0；分位数为线性插值。

汇总通过 fairness fingerprint 检查同模型不同精度的 token hash、GPU UUID、软件环境、attention、batch size、长度集合、warmup/repeats、model config 等相同。设置不同则拒绝合并，请指定独立 `--output-dir`。重复运行相同配置会将汇总指向最新完成结果，历史 run 保留，不将重跑样本混合统计。

## 9. 自动运行全部实验

```bash
bash scripts/run_all.sh

# 指定模型和 GPU，按模型、精度顺序运行。
DEVICE=cuda:2 bash scripts/run_all.sh \
  meta-llama/Meta-Llama-3-8B mistralai/Mistral-7B-v0.1

# 当前可用的两个精度；显式选择，不伪造 W4A4。
PRECISIONS='fp16 w8a8' DEVICE=cuda:2 bash scripts/run_all.sh
```

还支持 `PYTHON_BIN`、`OUTPUT_DIR`、`NUM_SAMPLES`、`WARMUP`、`REPEATS` 环境变量。默认仍尝试三个精度；失败会明确记录，继续其他配置，最终返回非零退出码。Python 根据物理 GPU UUID 加进程锁，防止该框架的多个实验占用同一 GPU；`CUDA_VISIBLE_DEVICES` 重映射也不能绕过。该锁不能约束其他用户的外部程序，正式实验应在独占 GPU 上运行。

## 10. 输出文件

| 路径（相对 `--output-dir`） | 内容 |
| --- | --- |
| `raw/runtime_samples.csv` | 每个模型/精度/长度/sample 一行，repeat median |
| `summary/runtime_by_length.csv` | 每配置的各类别、总时间和占比 |
| `summary/quantization_overhead.csv` | 单独的量化/转换类别及总和 |
| `summary/runtime_statistics.csv` | 每项指标的 mean/median/std/min/max/p10/p90 |
| `summary/figure1_data.csv` | Linear/Nonlinear/Other 及 compute fraction |
| `metadata/environment.json` | 当前汇总涉及的所有 run 的完整环境与 backend 证据 |
| `runs/<run_id>/environment.json` | 此次 run 的环境、数据来源、fairness fingerprint |
| `runs/<run_id>/repeats/L*.csv` | 未聚合的每次 repetition |
| `runs/<run_id>/raw/L*.csv` | 此次 run 的 sample 级结果 |
| `runs/<run_id>/kernels/*.json` | 每 repeat 的 kernel 名称、ATen 来源、调用次数、时长和 Event 诊断 |
| `runs/<run_id>/traces/*.json` | 默认每长度首个 sample/repeat 的完整 Chrome trace |
| `runs/<run_id>/status.json` | 完成/失败状态及已完成长度 |

使用 `--save-all-traces` 保留全部原始 trace，体积可能较大。CSV 在要求字段之外保留 provenance：`measurement_kind=measured`、`validation_status=passed`、`run_id`、`fairness_id` 和 token SHA256。单个配置完整测完且通过检查才进入汇总；失败配置不会产生替代数据。

## 11. 生成图

```bash
python plotting/plot_runtime_breakdown.py --input results/summary/figure1_data.csv
python plotting/plot_nonlinear_fraction.py --input results/summary/figure1_data.csv
python plotting/plot_cross_model.py --input results/summary/figure1_data.csv --seq-len 4096
```

输出 `figures/fig1a_runtime_breakdown.{pdf,png}`、`fig1b_nonlinear_fraction.{pdf,png}`、`fig1c_cross_model.{pdf,png}`。白底、少量网格、矢量 PDF、300 dpi PNG，多模型自动分面。图 (c) 至少需要两个模型。

当前 W4A4 缺失时，完整矩阵检查会拒绝绘图。可以显式使用 `--allow-incomplete` 绘制**不完整的真实测量矩阵**，缺失项标记 N/A 或断线，不填 0、不外推：

```bash
python plotting/plot_runtime_breakdown.py --allow-incomplete
python plotting/plot_nonlinear_fraction.py --allow-incomplete
```

支持 `--model MODEL`、`--output-dir` 和 `--seq-lens`。绘图检查 measured/validated 标记、输入一致性、总和与比例，不接受 projected/unresolved 行。不输出实验数值模板作为论文数据。

## 12. 测试与已知限制

```bash
python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES=2 RUN_CUDA_TESTS=1 RUN_FULL_LENGTH_TESTS=1 \
  python -m unittest discover -s tests -v
```

CUDA 测试使用 Hugging Face 的临时小模型 checkpoint，经过 `from_pretrained` 加载，覆盖两个模型族、两个可运行精度和所有要求长度，以及 integer padding、真实 INT8 kernel、分类可加性、完整 CLI/CSV 流程。测试中的人工 token 和绘图 fixtures 仅在临时目录，不是正式 WikiText-2 结果，也不支持论文结论。正式 7B/8B 实验需要下载对应权重与数据，独立执行上述命令。

当前限制：

- W4A4 backend 尚不可用，无法完成三精度论文矩阵；没有用其他精度替代。
- 单 GPU、batch 1、eager prefill；不支持 Tensor Parallel、offload、torch.compile、FlashAttention 或 fused SDPA。这些设置会改变统计口径。
- Python dispatch 与 CUPTI profiling 会扰动执行；physical runtime 使用独立未插桩 forward，但 kernel duration 仍是 profiling 条件下的测量。保留 trace 与诊断，不能声称没有 profiler 扰动。
- 不能分离的 fused quantization/GEMM 返回 `UNRESOLVED_FUSED_QUANT_GEMM`，不发布到 Figure 1。
- 未建立任何精度速度或非线性占比结论，必须由实际 checkpoint + WikiText-2 实验支持。

## 13. 代码组织及依据

根目录提供 `profile_llm.py`、`plotting/` 和 `scripts/run_all.sh`。核心包 `llm_runtime_profile/` 包含 `data/`、`models/`、`quantization/`、`instrumentation/`、`profiler/`，避免把本地目录命名为顶层 `datasets` 而遮蔽 Hugging Face Datasets。

计时和 trace 关联参考 [PyTorch profiler 官方说明](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)；attention backend 选择参考 [Transformers Attention Interface](https://huggingface.co/docs/transformers/attention_interface)。可核查的主要实现为 `profiler/trace_analysis.py`、`profiler/kernel_classifier.py` 和 `quantization/w8a8.py`。

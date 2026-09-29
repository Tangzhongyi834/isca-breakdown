# LLM Runtime Breakdown Profiling for Nonlinear Bottleneck Analysis

## 1. Project Objective

请实现一个基于 **Hugging Face Transformers + PyTorch + CUDA** 的 LLM inference profiling framework，用于研究：

> 随着 Transformer Linear 层从 FP16 降低到 W8A8 和 W4A4，低比特整数 GEMM 的执行时间不断降低时，非线性算子在整体计算时间中的占比如何随输入序列长度变化。

本实验主要服务于论文 Introduction 中的 **Nonlinear Bottleneck Shift** 分析。

实验不研究量化精度，因此：

- 不需要测试 perplexity；
- 不需要测试 downstream accuracy；
- 不需要比较不同 quantization algorithm 的精度；
- 不需要评估量化误差；
- 只分析 runtime；
- 所有论文结论必须来自真实 profiling 数据。

需要比较三种 Linear execution precision：

```text
FP16
W8A8
W4A4
```

支持的模型：

```text
LLaMA
Mistral
```

输入数据集：

```text
WikiText-2
```

测试序列长度：

```text
512
1024
2048
4096
```

---

# 2. Core Scientific Question

最终实验需要回答：

```text
How does increasingly aggressive low-bit linear computation
change the runtime composition of LLM inference across
different input sequence lengths?
```

重点观察：

```text
FP16 → W8A8 → W4A4
```

随着 Linear GEMM latency 降低：

```text
Linear Runtime Fraction ↓
```

以及：

```text
Nonlinear Runtime Fraction ↑
```

并分析该趋势如何随：

```text
Sequence Length:
512 → 1024 → 2048 → 4096
```

变化。

注意：

上述趋势只是研究假设。

**不得为了符合预期而调整实验数据。**

如果真实测量结果与预期不一致，应保留真实结果。

---

# 3. Mandatory Software Stack

必须使用：

```text
Python
PyTorch
CUDA
Hugging Face Transformers
Hugging Face Datasets
```

模型必须通过：

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(...)
```

加载。

禁止自行重写完整 Transformer 模型。

---

# 4. Supported Model Families

至少支持：

```text
LLaMA
Mistral
```

根据：

```python
model.config.model_type
```

自动判断：

```text
llama
mistral
```

不要 hard-code 到某一个具体模型 checkpoint。

代码应该支持例如：

```text
meta-llama/Llama-2-7b-hf
meta-llama/Meta-Llama-3-8B
mistralai/Mistral-7B-v0.1
```

模型名称必须通过命令行参数传入：

```bash
--model MODEL_NAME
```

---

# 5. Precision Modes

必须提供：

```bash
--precision fp16
--precision w8a8
--precision w4a4
```

---

# 6. FP16 Definition

FP16 作为 baseline。

模型：

```python
torch_dtype=torch.float16
```

Linear latency 定义为：

```text
FP16 GEMM execution time only
```

即只统计真正的矩阵乘法 kernel。

---

# 7. W8A8 Definition

W8A8 必须表示：

```text
INT8 Weight
+
INT8 Activation
+
real INT8 × INT8 GEMM
```

不能使用：

```text
INT8 Weight + FP16 Activation
```

冒充 W8A8。

优先考虑能够产生真实 INT8 GEMM 的 TorchAO / PyTorch CUDA backend。

运行时必须输出：

```text
Quantization Mode  : W8A8
Weight Precision   : INT8
Activation Precision: INT8
GEMM Backend       : ...
GEMM Kernel        : ...
```

---

# 8. W4A4 Definition

W4A4 必须严格表示：

```text
INT4 Weight
+
INT4 Activation
+
real INT4 × INT4 GEMM
```

禁止把以下方案标记成 W4A4：

```text
W4A16
INT4 weight-only
GPTQ W4A16
AWQ W4A16
bitsandbytes load_in_4bit
普通 INT4 weight + FP16 activation
```

必须提供统一 backend abstraction：

```python
class QuantBackend:
    def quantize_model(self, model):
        ...

    def validate(self):
        ...

    def backend_name(self):
        ...
```

并实现：

```text
FP16Backend
W8A8Backend
W4A4Backend
```

如果当前环境没有真实：

```text
INT4 × INT4 GEMM
```

则：

```text
DO NOT silently fallback to FP16.
DO NOT silently fallback to W4A16.
DO NOT label projected latency as measured W4A4.
```

应明确报错：

```text
True W4A4 CUDA backend is unavailable.
```

可以预留：

```bash
--allow-projected-w4a4
```

但默认：

```text
False
```

Projected 数据不得进入论文正式 measured results。

---

# 9. Critical Definition of Linear Latency

这是整个实验最重要的数据口径。

## Linear Latency

定义：

```text
Linear latency = GEMM computation latency only
```

即：

FP16：

```text
Linear = FP16 GEMM
```

W8A8：

```text
Linear = INT8 × INT8 GEMM
```

W4A4：

```text
Linear = INT4 × INT4 GEMM
```

---

# 10. Operations Excluded from Linear Latency

以下操作绝对不能加入：

```text
Linear latency
```

包括：

```text
Activation Quantization
Weight Quantization
Dequantization
Requantization
Scale Calculation
Scale Conversion
Zero-point Processing
FP → INT Conversion
INT → FP Conversion
Packing
Unpacking
Layout Conversion
Format Conversion
```

因此：

```text
Linear_W8A8 != Quantize + INT8 GEMM + Dequantize
```

而必须是：

```text
Linear_W8A8 = INT8 GEMM only
```

同理：

```text
Linear_W4A4 = INT4 GEMM only
```

---

# 11. Quantization Overhead

量化相关开销仍然可以测量，但必须单独存储。

定义：

```text
quantization_overhead_ms
```

包括：

```text
quantize
dequantize
requantize
scale
packing
unpacking
format conversion
```

这些数据仅用于：

```text
sanity check
implementation analysis
optional appendix
```

默认不进入论文 Figure 1 的 Linear 部分。

---

# 12. Runtime Categories

GPU runtime 需要划分为：

```text
1. Linear GEMM
2. Nonlinear
3. Attention MatMul
4. Quantization / Conversion Overhead
5. Other Compute
```

其中论文 Figure 1 使用：

```text
Linear
Nonlinear
Other Compute
```

Quantization / Conversion Overhead 不进入 Figure 1。

---

# 13. Linear Modules

Linear 主要包括：

## Attention Projection

```text
q_proj
k_proj
v_proj
o_proj
```

## FFN / MLP

```text
gate_proj
up_proj
down_proj
```

## LM Head

```text
lm_head
```

应该尽量基于 module type + architecture adapter 判断，而不是纯字符串硬编码。

量化之后，如果 backend 将：

```python
torch.nn.Linear
```

替换成自定义 quantized module，也必须保持对应关系。

---

# 14. Attention Matrix Multiplication

Attention 中：

```text
Q × K^T
```

以及：

```text
Attention Probability × V
```

必须单独归入：

```text
Attention MatMul
```

而不是：

```text
Linear
```

论文中的 Linear 特指：

```text
Projection GEMM
FFN GEMM
LM Head GEMM
```

---

# 15. Nonlinear Operators

必须分别统计：

```text
Softmax
RoPE
SiLU
RMSNorm
```

定义：

```text
Nonlinear =
Softmax
+ RoPE
+ SiLU
+ RMSNorm
```

输出：

```text
softmax_ms
rope_ms
silu_ms
rmsnorm_ms
nonlinear_ms
```

---

# 16. Softmax Profiling

为了能够独立统计 Softmax，不要默认使用会完全融合 attention 的实现。

Profiling 模式优先使用：

```python
attn_implementation="eager"
```

避免：

```text
FlashAttention
fully fused SDPA
```

将：

```text
QK MatMul
Softmax
PV MatMul
```

完全融合后无法分离。

只统计 attention 内 Softmax。

---

# 17. RoPE Profiling

对于 LLaMA / Mistral：

不能只统计：

```text
rotary_emb()
```

还必须考虑：

```text
apply_rotary_pos_emb(...)
```

真正对 Q/K 进行：

```text
sin/cos
rotation
element-wise transform
```

的部分。

因此 RoPE latency 应覆盖完整 rotary application，但不包含：

```text
q_proj
k_proj
```

等 Linear GEMM。

---

# 18. SiLU Profiling

LLaMA / Mistral FFN 通常为：

```text
gate_proj
→ SiLU
→ element-wise multiply
→ down_proj
```

只统计：

```text
SiLU activation
```

不要把：

```text
gate_proj
up_proj
down_proj
```

计入 SiLU。

也不要把：

```text
gate × up
```

element-wise multiply 自动算进 SiLU。

---

# 19. RMSNorm Profiling

识别：

```text
LlamaRMSNorm
MistralRMSNorm
```

将完整 RMSNorm 作为一个 nonlinear operator。

统计其中：

```text
square
reduction
rsqrt
scale
```

的完整执行时间。

---

# 20. Dataset

最终论文实验必须使用：

```text
WikiText-2
```

加载：

```python
from datasets import load_dataset

dataset = load_dataset(
    "Salesforce/wikitext",
    "wikitext-2-raw-v1",
    split="test",
)
```

使用：

```text
test split
```

不要使用 synthetic random token 作为最终论文数据。

---

# 21. WikiText-2 Preprocessing

WikiText-2 中存在：

```text
empty lines
section titles
paragraph boundaries
```

过滤完全为空的文本：

```python
texts = [
    example["text"]
    for example in dataset
    if example["text"].strip()
]
```

然后按照原始顺序拼接：

```python
full_text = "\n\n".join(texts)
```

不要：

```text
shuffle
lowercase
remove punctuation
rewrite text
```

保持原始 WikiText-2 内容。

---

# 22. Tokenization

每个模型必须使用自己的 tokenizer：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=True,
)
```

然后：

```python
encoded = tokenizer(
    full_text,
    add_special_tokens=False,
    return_attention_mask=False,
)

all_tokens = encoded["input_ids"]
```

注意：

```text
LLaMA
Mistral
```

可以产生不同 tokenization。

这是允许的。

但对于同一个模型：

```text
FP16
W8A8
W4A4
```

必须使用完全相同的 token IDs。

---

# 23. Sequence Lengths

必须测试：

```text
512
1024
2048
4096
```

命令行支持：

```bash
--seq-lens 512 1024 2048 4096
```

---

# 24. Fair Sample Construction

不要针对：

```text
512
1024
2048
4096
```

分别随机选择不同文本。

必须先构造：

```text
4096-token base window
```

然后：

```python
base = all_tokens[start:start + 4096]

sample_512 = base[:512]
sample_1024 = base[:1024]
sample_2048 = base[:2048]
sample_4096 = base[:4096]
```

因此：

```text
sample_512
⊂ sample_1024
⊂ sample_2048
⊂ sample_4096
```

这样同一个 sample 在不同 sequence length 下具有完全一致的文本前缀。

---

# 25. Number of Samples

默认：

```text
num_samples = 32
```

每一个：

```text
Model
× Precision
× Sequence Length
```

都测试相同的 32 个 WikiText-2 base windows。

例如：

```text
LLaMA
FP16
512
```

使用 sample 0–31。

那么：

```text
LLaMA
W8A8
512
```

以及：

```text
LLaMA
W4A4
512
```

也必须使用完全相同的 sample 0–31。

---

# 26. Base Window Generation

使用 deterministic non-overlapping windows：

```python
MAX_SEQ_LEN = 4096
NUM_SAMPLES = 32

samples = []

for i in range(NUM_SAMPLES):
    start = i * MAX_SEQ_LEN
    end = start + MAX_SEQ_LEN

    if end > len(all_tokens):
        break

    samples.append(
        all_tokens[start:end]
    )
```

允许提供：

```bash
--token-offset
```

默认：

```text
token_offset = 0
```

如果指定：

```python
start = token_offset + i * MAX_SEQ_LEN
```

默认论文实验不要随机采样。

---

# 27. Dataset Cache

为了保证所有 precision 使用相同 token IDs，必须缓存 tokenized samples。

建议：

```text
data/
└── wikitext2/
    ├── llama/
    │   └── samples.pt
    └── mistral/
        └── samples.pt
```

保存：

```python
{
    "dataset": "Salesforce/wikitext",
    "config": "wikitext-2-raw-v1",
    "split": "test",
    "model": model_name,
    "num_samples": 32,
    "max_seq_len": 4096,
    "samples": samples,
}
```

第一次：

```text
download
→ tokenize
→ construct samples
→ cache
```

之后：

```text
load cached samples
```

---

# 28. Input Tensor

对于每个 sequence length：

```python
input_ids = torch.tensor(
    base_tokens[:seq_len],
    dtype=torch.long,
    device=device,
).unsqueeze(0)
```

必须保证：

```python
assert input_ids.shape == (1, seq_len)
```

Batch size 默认：

```text
1
```

不要 padding。

---

# 29. Special Tokens

序列长度：

```text
512
1024
2048
4096
```

必须表示真实：

```text
model input token count
```

因此：

```python
add_special_tokens=False
```

避免：

```text
4096 text tokens
+
1 BOS
=
4097 tokens
```

---

# 30. Main Benchmark Phase

论文 Figure 1 主要测试：

```text
Prefill
```

默认：

```python
with torch.inference_mode():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
```

Figure 1 中统一：

```text
use_cache = False
```

用于分析纯 prefill compute runtime composition。

如果未来需要 decode：

```text
use_cache=True
```

必须作为另一组独立实验。

---

# 31. Why Prefill

本实验研究：

```text
runtime composition vs sequence length
```

Prefill 更适合观察这种变化。

随着长度：

```text
L
```

变化：

```text
Linear Projection / FFN
~ O(L)

SiLU
~ O(L)

RMSNorm
~ O(L)

RoPE
~ O(L)

Attention score computation
~ O(L^2)

Softmax attention workload
grows strongly with L
```

因此：

```text
512 → 1024 → 2048 → 4096
```

能够展示 runtime composition 如何随 context length 变化。

---

# 32. CUDA Timing Method

禁止使用：

```python
time.time()
```

统计 GPU kernel latency。

必须使用：

```python
torch.cuda.Event(enable_timing=True)
```

示例：

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

torch.cuda.synchronize()

start.record()

operation()

end.record()

torch.cuda.synchronize()

latency_ms = start.elapsed_time(end)
```

---

# 33. Avoid Synchronization Distortion

不能在每个 operator 后调用：

```python
torch.cuda.synchronize()
```

这样会严重破坏正常 GPU pipeline。

推荐：

```text
record CUDA event
↓
run operators
↓
record CUDA event
↓
complete forward
↓
single synchronize
↓
calculate elapsed times
```

---

# 34. CUDA Timer Infrastructure

实现：

```python
class CUDAEventTimer:
    ...
```

支持：

```python
timer.start("softmax")

operator()

timer.stop("softmax")
```

同一类 operator 可以被多个 Transformer layer 重复调用。

保存：

```python
{
    "linear": [...],
    "softmax": [...],
    "rope": [...],
    "silu": [...],
    "rmsnorm": [...],
    "attention_matmul": [...],
}
```

最后聚合所有 layer。

---

# 35. Critical Kernel-Level Linear Profiling

对于 W8A8/W4A4，如果 high-level Linear module 的执行过程为：

```text
Quantize
↓
INT GEMM
↓
Dequantize
```

不能直接 profile：

```python
linear_module.forward()
```

否则得到的是：

```text
Quantize + GEMM + Dequantize
```

这不符合实验定义。

必须尽可能深入 backend。

理想形式：

```python
with timer("quantize"):
    x_int = quantize(x)

with timer("linear_gemm"):
    y_int = int_gemm(
        x_int,
        weight_int
    )

with timer("dequantize"):
    y = dequantize(y_int)
```

最终：

```text
linear_ms
=
linear_gemm only
```

---

# 36. Fused Quantization + GEMM Kernel

如果 backend 使用：

```text
Quantize + GEMM
```

完全融合的 kernel，并且无法拆开：

不能直接把整个 fused kernel 标记为：

```text
INT GEMM latency
```

此时：

1. 检查 backend 是否支持更低层 kernel timing；
2. 使用 PyTorch profiler；
3. 必要时使用 Nsight Systems / Nsight Compute；
4. 尝试识别真正 GEMM kernel；
5. 如果仍无法可靠分离，则标记：

```text
UNRESOLVED_FUSED_QUANT_GEMM
```

该配置不得进入论文 Figure 1。

---

# 37. Two Runtime Definitions

必须同时记录：

## Physical Runtime

```text
physical_total_ms
```

表示真实 wall-clock GPU execution：

```text
Linear
+ Nonlinear
+ Attention MatMul
+ Quantization
+ Conversion
+ Other
```

用于 sanity check。

---

## Compute Runtime

论文 Figure 1 使用：

```text
compute_total_ms
```

定义：

```text
compute_total_ms
=
linear_ms
+ nonlinear_ms
+ attention_matmul_ms
+ other_compute_ms
```

明确排除：

```text
quantization_overhead_ms
```

因此：

```text
compute_total_ms != physical_total_ms
```

这是刻意设计的。

论文研究的是：

```text
low-bit integer compute itself
```

对 runtime composition 的影响，而不是 quantizer 软件开销。

---

# 38. Runtime Fraction Definition

Linear fraction：

```python
linear_pct = (
    linear_ms
    / compute_total_ms
    * 100
)
```

Nonlinear fraction：

```python
nonlinear_pct = (
    nonlinear_ms
    / compute_total_ms
    * 100
)
```

Other fraction：

```python
other_pct = (
    attention_matmul_ms
    + other_compute_ms
) / compute_total_ms * 100
```

必须满足：

```text
Linear %
+ Nonlinear %
+ Other %
≈ 100%
```

---

# 39. Avoid Double Counting

以下分类必须互斥：

```text
Linear
Softmax
RoPE
SiLU
RMSNorm
Attention MatMul
Other Compute
```

不要将一个 kernel 同时统计到两个 category。

特别注意：

```text
QK^T
```

不能既进入：

```text
Linear
```

又进入：

```text
Attention MatMul
```

---

# 40. Warmup

每个：

```text
Model
× Precision
× Sequence Length
```

都要单独 warmup。

默认：

```text
warmup = 5
```

使用：

```text
sample 0
```

作为 warmup。

Warmup 不进入正式结果。

---

# 41. Repetition Strategy

默认：

```text
num_samples = 32
repeats_per_sample = 3
```

对于每个 sample：

```text
run 3 times
```

得到三次 latency 后取：

```text
median
```

作为该 sample 的结果。

之后在 32 个 sample 上计算：

```text
mean
median
std
min
max
p10
p90
```

论文默认使用：

```text
median across samples
```

除非后续明确修改。

---

# 42. Full Experimental Matrix

必须支持自动运行：

```text
Models:
    LLaMA
    Mistral

Precisions:
    FP16
    W8A8
    W4A4

Sequence Lengths:
    512
    1024
    2048
    4096
```

因此每个模型：

```text
3 × 4 = 12
```

组主要 configuration。

每组：

```text
32 WikiText-2 samples
```

---

# 43. Recommended Execution Order

不要在一个 iteration 内频繁切换 precision。

推荐：

```text
Load model
↓
Configure FP16
↓
512
1024
2048
4096
↓
Release model

Load model
↓
Configure W8A8
↓
512
1024
2048
4096
↓
Release model

Load model
↓
Configure W4A4
↓
512
1024
2048
4096
↓
Release model
```

---

# 44. CLI

主程序：

```text
profile_llm.py
```

示例：

```bash
python profile_llm.py \
    --model meta-llama/Llama-2-7b-hf \
    --precision fp16 \
    --dataset Salesforce/wikitext \
    --dataset-config wikitext-2-raw-v1 \
    --split test \
    --seq-lens 512 1024 2048 4096 \
    --num-samples 32 \
    --batch-size 1 \
    --warmup 5 \
    --repeats 3 \
    --phase prefill \
    --device cuda:0
```

W8A8：

```bash
python profile_llm.py \
    --model meta-llama/Llama-2-7b-hf \
    --precision w8a8 \
    --seq-lens 512 1024 2048 4096 \
    --num-samples 32 \
    --phase prefill \
    --device cuda:0
```

Mistral：

```bash
python profile_llm.py \
    --model mistralai/Mistral-7B-v0.1 \
    --precision w8a8 \
    --seq-lens 512 1024 2048 4096 \
    --num-samples 32 \
    --phase prefill \
    --device cuda:0
```

---

# 45. Batch Script

实现：

```text
scripts/run_all.sh
```

支持自动运行全部：

```text
Model
× Precision
× Sequence Length
```

禁止多个 experiment 同时占用同一个 GPU。

---

# 46. Code Architecture

建议目录：

```text
llm_runtime_profile/
│
├── profile_llm.py
│
├── README.md
├── requirements.txt
│
├── datasets/
│   ├── __init__.py
│   └── wikitext2.py
│
├── profiler/
│   ├── __init__.py
│   ├── cuda_timer.py
│   ├── runtime_profiler.py
│   ├── kernel_classifier.py
│   └── result_aggregator.py
│
├── models/
│   ├── __init__.py
│   ├── base_adapter.py
│   ├── llama_adapter.py
│   └── mistral_adapter.py
│
├── quantization/
│   ├── __init__.py
│   ├── base.py
│   ├── fp16.py
│   ├── w8a8.py
│   └── w4a4.py
│
├── instrumentation/
│   ├── __init__.py
│   ├── linear_gemm.py
│   ├── quantization.py
│   ├── attention_matmul.py
│   ├── softmax.py
│   ├── rope.py
│   ├── silu.py
│   └── rmsnorm.py
│
├── plotting/
│   ├── plot_runtime_breakdown.py
│   ├── plot_nonlinear_fraction.py
│   └── plot_cross_model.py
│
├── scripts/
│   └── run_all.sh
│
├── data/
│   └── wikitext2/
│
└── results/
```

---

# 47. Model Adapter

定义：

```python
class ModelAdapter:
    def get_linear_modules(self):
        ...

    def get_rmsnorm_modules(self):
        ...

    def instrument_rope(self):
        ...

    def instrument_silu(self):
        ...

    def instrument_softmax(self):
        ...

    def instrument_attention_matmul(self):
        ...
```

实现：

```python
class LlamaAdapter(ModelAdapter):
    ...

class MistralAdapter(ModelAdapter):
    ...
```

不要在核心 profiler 中到处出现：

```python
if model_type == "llama":
```

architecture-specific logic 必须封装。

---

# 48. Raw Output

每一个 WikiText-2 sample 保存一行：

```csv
model,precision,seq_len,sample_id,linear_ms,softmax_ms,rope_ms,silu_ms,rmsnorm_ms,nonlinear_ms,attention_matmul_ms,quantization_overhead_ms,other_compute_ms,compute_total_ms,physical_total_ms
```

例如：

```text
Llama-2-7B,FP16,512,0,...
Llama-2-7B,FP16,512,1,...
...
Llama-2-7B,W8A8,512,0,...
...
Llama-2-7B,W4A4,4096,31,...
```

---

# 49. Aggregated Output

生成：

```text
results/summary/runtime_by_length.csv
```

格式：

```csv
model,precision,seq_len,num_samples,linear_ms,softmax_ms,rope_ms,silu_ms,rmsnorm_ms,nonlinear_ms,attention_matmul_ms,other_compute_ms,compute_total_ms,physical_total_ms,linear_pct,nonlinear_pct,other_pct
```

---

# 50. Quantization Overhead Output

单独生成：

```text
results/summary/quantization_overhead.csv
```

格式：

```csv
model,precision,seq_len,quantize_ms,dequantize_ms,requantize_ms,scale_ms,packing_ms,total_quantization_overhead_ms
```

该文件默认不用于 Figure 1。

---

# 51. Environment Metadata

自动保存：

```text
results/metadata/environment.json
```

必须记录：

```text
GPU name
GPU compute capability
CUDA version
NVIDIA driver
PyTorch version
Transformers version
TorchAO version
Python version
Model name
Model type
Precision
Quantization backend
GEMM backend
Attention implementation
Batch size
Sequence length
Number of samples
Warmup count
Repeat count
use_cache
```

---

# 52. Runtime Validation Print

每个实验完成后输出：

```text
========================================
Experiment
========================================

Model              : ...
Precision          : ...
Sequence Length    : ...
Samples            : 32

Linear GEMM        : XX.XX ms

Nonlinear          : XX.XX ms
    Softmax        : XX.XX ms
    RoPE           : XX.XX ms
    SiLU           : XX.XX ms
    RMSNorm        : XX.XX ms

Attention MatMul   : XX.XX ms
Other Compute      : XX.XX ms

Quantization       : XX.XX ms

Compute Total      : XX.XX ms
Physical Total     : XX.XX ms

Linear Fraction    : XX.XX %
Nonlinear Fraction : XX.XX %
Other Fraction     : XX.XX %
========================================
```

---

# 53. Precision Validation

W8A8 必须打印：

```text
Weight Precision     : INT8
Activation Precision : INT8
GEMM Kernel          : ...
Backend              : ...
```

W4A4 必须打印：

```text
Weight Precision     : INT4
Activation Precision : INT4
GEMM Kernel          : ...
Backend              : ...
```

如果无法证明实际 low-bit kernel，则：

```text
DO NOT generate final paper data.
```

---

# 54. Figure 1(a): Runtime Composition

生成：

```text
plot_runtime_breakdown.py
```

Figure 1(a) 展示：

```text
Runtime Composition vs Sequence Length
```

横轴分成：

```text
512
1024
2048
4096
```

每个 sequence length 包含：

```text
FP16
W8A8
W4A4
```

即：

```text
           512              1024
      FP16 W8A8 W4A4   FP16 W8A8 W4A4

          2048              4096
      FP16 W8A8 W4A4   FP16 W8A8 W4A4
```

每根柱子使用：

```text
100% stacked bar
```

包括：

```text
Linear
Nonlinear
Other
```

纵轴：

```text
Compute Runtime Fraction (%)
```

而不是：

```text
End-to-End Wall-Clock Fraction
```

因为 quantization overhead 被明确排除。

---

# 55. Figure 1(b): Nonlinear Fraction vs Sequence Length

生成：

```text
plot_nonlinear_fraction.py
```

横轴：

```text
Sequence Length
512
1024
2048
4096
```

纵轴：

```text
Nonlinear Runtime Fraction (%)
```

三条曲线：

```text
FP16
W8A8
W4A4
```

所有数据点必须来自：

```text
runtime_by_length.csv
```

中的真实 measurement。

禁止：

```text
theoretical projection
Amdahl's-law derived points
manual data
```

---

# 56. Optional Figure 1(c): Cross-Model Comparison

可额外生成：

```text
plot_cross_model.py
```

用于比较：

```text
LLaMA
Mistral
```

可以选择：

```text
sequence length = 4096
```

横轴：

```text
FP16
W8A8
W4A4
```

纵轴：

```text
Nonlinear Runtime Fraction (%)
```

用于回答：

```text
Is the bottleneck shift model-specific?
```

---

# 57. Figure Data File

生成：

```text
results/summary/figure1_data.csv
```

格式：

```csv
model,precision,seq_len,linear_ms,nonlinear_ms,other_ms,compute_total_ms,linear_pct,nonlinear_pct,other_pct
```

Plotting script 不得写死任何实验数字。

---

# 58. Plotting Requirements

使用：

```text
matplotlib
```

保持论文风格：

- white background；
- minimal grid；
- readable labels；
- no 3D chart；
- no unnecessary decoration；
- publication-quality PDF；
- 同时保存 PNG；
- PDF 使用 vector format。

输出：

```text
figures/
├── fig1a_runtime_breakdown.pdf
├── fig1a_runtime_breakdown.png
├── fig1b_nonlinear_fraction.pdf
├── fig1b_nonlinear_fraction.png
├── fig1c_cross_model.pdf
└── fig1c_cross_model.png
```

---

# 59. Fairness Requirements

对于同一个 model：

```text
FP16
W8A8
W4A4
```

必须共享：

```text
same WikiText-2 samples
same token IDs
same batch size
same sequence lengths
same attention implementation
same GPU
same use_cache configuration
same warmup method
same repetition method
same profiler
```

唯一主要变量：

```text
Linear precision / GEMM kernel
```

---

# 60. Sanity Checks

正式运行前必须检查：

```python
assert input_ids.shape[0] == 1
assert input_ids.shape[1] == seq_len
assert torch.isfinite(outputs.logits).all()
```

同时检查：

```text
model device
input device
dtype
output shape
```

不需要比较量化模型和 FP16 logits 的 numerical accuracy。

---

# 61. Runtime Consistency Check

需要检查：

```text
Linear
+ Nonlinear
+ Attention MatMul
+ Other Compute
≈ Compute Total
```

如果误差超过：

```text
5%
```

打印 warning：

```text
WARNING:
Measured operator categories differ from compute total by XX%.
Profiling may contain missing or double-counted kernels.
```

---

# 62. Dataset Sanity Print

启动时打印：

```text
Dataset            : Salesforce/wikitext
Dataset Config     : wikitext-2-raw-v1
Split              : test
Tokenizer          : ...
Number of Samples  : 32
Max Sequence Length: 4096
```

并输出：

```text
Sample 0:
512  -> 512 tokens
1024 -> 1024 tokens
2048 -> 2048 tokens
4096 -> 4096 tokens
```

---

# 63. Implementation Milestones

不要一次性把所有功能混在一起实现。

## Milestone 1

完成：

```text
LLaMA
+
FP16
+
WikiText-2
```

准确得到：

```text
Linear
Softmax
RoPE
SiLU
RMSNorm
Attention MatMul
Other
```

验证：

```text
sum(categories) ≈ compute_total
```

---

## Milestone 2

加入：

```text
Mistral
```

验证 ModelAdapter abstraction。

---

## Milestone 3

加入：

```text
W8A8
```

确保：

```text
INT8 × INT8 GEMM
```

是真实执行。

确保：

```text
Linear timing = INT8 GEMM only
```

---

## Milestone 4

加入：

```text
W4A4
```

只有真实：

```text
INT4 × INT4 GEMM
```

能够运行时才进入论文实验。

---

## Milestone 5

运行完整：

```text
Model
× Precision
× Sequence Length
× WikiText Samples
```

生成全部 CSV。

---

## Milestone 6

自动生成 Figure 1。

---

# 64. README Requirements

README 必须包含：

```text
1. Project purpose
2. Environment installation
3. Supported models
4. Supported precision modes
5. WikiText-2 preparation
6. FP16 profiling
7. W8A8 profiling
8. W4A4 backend requirement
9. Running all experiments
10. Output CSV descriptions
11. Generating figures
12. Known backend limitations
13. Definition of Linear latency
```

必须特别强调：

```text
Linear latency means GEMM kernel latency only.
Quantization/dequantization overhead is excluded.
```

以及：

```text
W4A4 means true INT4 activation × INT4 weight GEMM.
Weight-only INT4 is not accepted.
```

---

# 65. Final Acceptance Criteria

运行：

```bash
python profile_llm.py \
    --model meta-llama/Llama-2-7b-hf \
    --precision fp16 \
    --seq-lens 512 1024 2048 4096 \
    --num-samples 32 \
    --phase prefill \
    --device cuda:0
```

能够生成真实 WikiText-2 profiling 数据。

然后：

```bash
python profile_llm.py \
    --model meta-llama/Llama-2-7b-hf \
    --precision w8a8 \
    --seq-lens 512 1024 2048 4096 \
    --num-samples 32 \
    --phase prefill \
    --device cuda:0
```

也必须运行成功。

Mistral 同样支持。

当真实 W4A4 backend 可用时：

```bash
python profile_llm.py \
    --model meta-llama/Llama-2-7b-hf \
    --precision w4a4 \
    --seq-lens 512 1024 2048 4096 \
    --num-samples 32 \
    --phase prefill \
    --device cuda:0
```

也必须运行成功。

之后：

```bash
python plotting/plot_runtime_breakdown.py \
    --input results/summary/figure1_data.csv
```

生成 Figure 1(a)。

以及：

```bash
python plotting/plot_nonlinear_fraction.py \
    --input results/summary/figure1_data.csv
```

生成 Figure 1(b)。

---

# 66. Final Scientific Output

最终需要获得如下二维实验矩阵：

```text
                     Sequence Length

                512   1024   2048   4096

FP16            XX%    XX%    XX%    XX%
W8A8            XX%    XX%    XX%    XX%
W4A4            XX%    XX%    XX%    XX%
```

其中每一个值表示：

```text
Nonlinear Runtime Fraction
```

并且每一个值都必须来自：

```text
real WikiText-2 inference
+
real CUDA profiling
```

最终论文希望分析：

```text
Lower Linear Precision
        +
Longer Sequence Length
        ↓
Changing Runtime Composition
        ↓
Emerging Nonlinear Bottleneck
```

但所有结论都必须由真实数据决定。

---

# 67. Most Important Rules

请始终遵守以下五条规则：

### Rule 1

```text
Linear latency = GEMM only.
```

### Rule 2

```text
Quantization / dequantization is NOT Linear latency.
```

### Rule 3

```text
W4A4 must be real INT4 × INT4 compute.
```

### Rule 4

```text
Different precisions must use exactly the same WikiText-2 token samples.
```

### Rule 5

```text
Never fabricate or manually adjust profiling results.
```
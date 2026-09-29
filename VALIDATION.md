# 实现验收记录

日期：2026-09-25。此记录区分代码验收与正式论文实验，不提供性能趋势结论。

## million 环境

- Python：`/home/tangzhongyi/miniconda3/envs/million/bin/python`
- PyTorch：2.5.1+cu121
- Transformers：4.46.3
- Datasets：2.18.0
- Matplotlib：3.9.2
- GPU：NVIDIA H20-3e，SM90；测试选择物理 GPU 2。

没有为验收升级或替换 `million` 环境中的现有依赖。

## 自动化测试

执行：

```bash
conda activate million
CUDA_VISIBLE_DEVICES=2 RUN_CUDA_TESTS=1 RUN_FULL_LENGTH_TESTS=1 \
  python -m unittest discover -s tests -v
```

结果：13 个测试全部通过，无跳过。覆盖：

- WikiText-2 预处理、窗口前缀、offset、缓存隔离和损坏检测。
- 互斥算子归类、CUPTI 关联、GEMM workspace memset 单独计入转换。
- Repeat/sample 两级 median、派生总和、比例及统计量。
- LLaMA 和 Mistral 的 HF 临时 checkpoint 加载及 FP16/W8A8 CUDA profiling。
- 两个模型族、两个精度分别验证 16、512、1024、2048、4096 长度；16 用于覆盖 padding。
- 真正 INT8×INT8→INT32 GEMM、非对齐矩阵形状、输出有限性。
- CLI 到 CSV/metadata 的完整流程，重跑不重复累计。
- 三种图的 PDF/PNG 输出，拒绝 projected 数据，明确处理缺失 W4A4。
- W4A4 无合格 backend 时失败，不回退。

这些测试的随机小模型、人工 token 和绘图 fixtures 只存在于临时目录，不作为论文数据。

## 真实 WikiText-2 入口验证

额外在 `million` 环境执行了真实下载/分词/缓存/inference/profiling：

- Checkpoint：`HuggingFaceH4/tiny-random-LlamaForCausalLM`。
- 数据：`Salesforce/wikitext` / `wikitext-2-raw-v1` / `test`。
- 精度：FP16、W8A8。
- 序列长度：512、1024、2048（该 checkpoint 声明的 context 上限为 2048）。
- 每配置 1 个 sample、1 次 warmup、1 次 repeat，合计 6 个配置。
- 两种精度的缓存 token SHA256 均为 `a47765532401a8f118bdfbfc040b3f36c2767b7db7e6b94b81874bafc1494878`。
- W8A8 捕获到 `cutlass_80_tensorop_i16832gemm_s8_...` CUDA kernel，并通过 INT32 结果校验。

输出留在临时验证目录：

```text
/tmp/isca-wikitext-validation-n3fxg0/million-data/
/tmp/isca-wikitext-validation-n3fxg0/million-results/
/tmp/isca-wikitext-validation-n3fxg0/million-figures/
```

另在 PyTorch 2.6.0 / Transformers 4.48.3 环境完成了相同入口验证。不同软件环境的结果保存在不同目录，没有混合汇总。

## 尚未执行的正式实验

尚未运行 LLaMA 7B/8B、Mistral 7B 的完整 32-sample、3-repeat 论文矩阵。小型随机 checkpoint 的结果只验证软件管线；共享服务器上的功能验收也不能替代独占 GPU 上的正式测量。

当前没有集成可验证的真实 W4A4 backend，`--precision w4a4` 返回退出码 2 和 `True W4A4 CUDA backend is unavailable.`。不存在 W4A16 替代值、理论外推值或完整三精度论文结论。

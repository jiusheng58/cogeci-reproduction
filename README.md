# CogECI 复现：文档级事件因果识别

机器智能课程设计 · 第 4 个项目 · 第六组（张旭、路文博）

本仓库是对 **CogECI** 的复现工作，包含官方代码的补全与修补、伪标签生成工具链、
训练与评测脚本、完整实验日志，以及 LaTeX 报告源文件。

- 官方代码：<https://github.com/zefanZhang-cn/CogECI>
- 数据集：EventStoryLine v0.9（ESC）

## 主要结果

ESC，因果存在性二分类口径，5 折交叉验证：

| 模型 | 整体 F1 | 句内 F1 | 跨句 F1 |
|---|---|---|---|
| iLIF（原文报告） | 62.5 | 71.2 | 59.1 |
| **本复现（CogECI）** | **64.9** | **77.4** | 58.8 |

5 折明细：`P=0.6812`、`R=0.6225`、`F1=0.6493 ± 0.0261`。

## 本仓库相对官方仓库的改动

官方仓库发布时缺失 4 类文件，其中 `utils.py` 被 `main.py` 直接 import，缺失即无法启动；
伪标签 CSV 缺失时代码会**静默回退为全零标签**（训练照常跑但学不到东西）。

| 文件 | 类型 | 内容 |
|---|---|---|
| `_repo/model/model.py` | 性能 | 5 处向量化改写，实测加速 12–17×，已验证数值等价 |
| `_repo/model/model.py` | 功能 | CGL 权重 `w=6` 硬编码 → `--cgl_w`（默认 6.0） |
| `_repo/model/model.py` | 修复 | 删除死代码 `true_sent`；修复 `sent_ground` 重复 append |
| `_repo/parameter.py` | 功能 | 新增 `--grad_ckpt`、`--cgl_w` |
| `_repo/main.py` | 功能 | 每折按验证 F1 保存权重到 `checkpoints/<method>/` |
| `_repo/main.py` | 兼容 | 日志目录 `seed:{s}` → `seed_{s}`（Windows 不允许冒号） |
| `_repo/utils.py` | 补齐 | 取自 iLIF 仓库 |

数值等价性验证见 `tools/equiv_patch.py`：loss 偏差为 0、预测偏差 <1.2e-07、
梯度范数偏差 <1e-08、图边选取 `torch.equal` 为真。

## 伪标签说明

CogECI 的第一阶段本应由 LLM 为每个因果正例事件对标注支撑句。本项目在打标进行到
第 1752 次时平台免费额度耗尽，改用**规则法离线构造**伪标签，并用已完成的
545 条 Qwen3.8-27B 标注做一致性检验：

| 对比 | 句级 P | 句级 R | 句级 F1 | 规则集 ⊆ LLM 集 |
|---|---|---|---|---|
| 规则 strict vs. LLM strict | **0.985** | 0.753 | 0.853 | 97.4% |

规则设计：句内对取事件所在句；句间对取区间（超过 8 句则退化为局部窗口）。
`_repo/data/examples/llm/` 下四份 CSV 即最终用于训练的伪标签，
`ref_llm27b/` 为 LLM 标注（仅作对照，不进训练），`raw_*.jsonl` 为全部调用的提示词与原始响应留痕。

## 复现步骤

### 1. 环境

```bash
python -m venv .venv
.venv/Scripts/pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/pip install torch_geometric==2.5.3 transformers==4.46.3 \
    scikit-learn tqdm networkx openai
```

> `transformers` 必须为 4.x —— 5.x 已移除 `AdamW`，而 `main.py` 依赖它。

### 2. 补齐依赖文件（本仓库未包含，体积原因）

- `_repo/utils.py`：本仓库**已包含**
- `_repo/data/database/exist/*.pkl`：取自 iLIF 仓库
- `_repo/model/bert-base-uncased/`：从 HuggingFace 下载
- `_repo/checkpoints/`：训练产物，随课程提交包提供（5 折各约 535 MB）

### 3. 生成伪标签并终检

```bash
for S in dev train; do for V in strict loose; do
  ./.venv/Scripts/python.exe tools/pseudo_from_rule.py --split $S --variant $V \
      --out_csv _repo/data/examples/llm/rule/${S}_${V}.csv
done; done

./.venv/Scripts/python.exe tools/validate_csv.py \
    --pkl _repo/data/database/exist/EventStoryLine_bert_intra_and_inter_train_features.pkl \
    --csv _repo/data/examples/llm/llama3help_train_sen.csv
```

> `validate_csv.py` 是必需的：它检查 doc_id 唯一、answer 条数等于事件对数、
> 编号不越界、覆盖全部文档。缺任何一篇都会让标签矩阵与事件对不匹配。

### 4. 训练

必须在 `_repo/` 目录下运行（`process_esc_csr.py` 用相对路径，否则读不到伪标签）。

```bash
cd _repo
../.venv/Scripts/python.exe -u main.py --dataset_type EventStoryLine \
  --cache_path ./data/database/exist --model_name ./model/bert-base-uncased \
  --no_of_classes 2 --t_lr 2e-5 --lr 1e-4 --w 0.6 --threshold 2 --epoch 30 \
  --batch_size 1 --beta_intra 0.3 --beta_inter 0.7 --max_iteration 9 \
  --min_iteration 2 --num_heads 4 --mlp_drop 0.4 --seed 209 --gamma 2.0 \
  --grad_ckpt --cgl_w 6.0 --method E1_full_rulelabels
```

消融：`--cgl_w 0`（E0，去 CGL）/ `--cgl_w 3`（E2），`--method` 相应改名。

### 5. 辅助脚本

```bash
./.venv/Scripts/python.exe tools/agree_rule_llm.py --split dev      # 规则 vs LLM 一致性
./.venv/Scripts/python.exe tools/agree_by_reltype.py --split dev    # 按 intra/inter 拆分
./.venv/Scripts/python.exe tools/equiv_patch.py                     # 补丁数值等价性验证
./.venv/Scripts/python.exe tools/bench_vram.py                      # 显存峰值实测
```

## 硬件与耗时

全部实验在 RTX 4050 Laptop（6 GB）上完成。默认配置峰值显存 6824 MB（溢出），
开启 `--grad_ckpt` 后降至 2483 MB；配合 5 处向量化补丁，单折训练约 50 分钟，5 折约 4.2 小时。
若显存 ≥16 GB，可去掉 `--grad_ckpt` 获得更快单步速度，数值结果不变。

## 报告

LaTeX 源码在 `report/`，编译：

```bash
cd report && xelatex report.tex && xelatex report.tex
```

## 致谢

`utils.py` 与 ESC 特征缓存来自 iLIF 作者的开源仓库，特此致谢。
本仓库仅用于课程学习与复现研究。

## 引用

```bibtex
@inproceedings{liu-etal-2024-identifying,
  title  = {Identifying while Learning for Document Event Causality Identification},
  author = {Liu, Cheng and Xiang, Wei and Wang, Bang},
  year   = {2024},
  note   = {ACL 2024, arXiv:2405.20608}
}
```

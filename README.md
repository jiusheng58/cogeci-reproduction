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

> **增益应归因于 SGM 与 DSCRM，而非 CGL。** 见下方「重要发现」。

## 重要发现：官方实现的 CGL 项没有梯度

`_repo/model/model.py` 中，上下文对齐损失（CGL）在计算前对余弦相似度做了硬阈值二值化：

```python
cosine_sim = F.cosine_similarity(event_expanded, sentence_expanded, dim=2)
flag = float(1 / cosine_sim.size(1))
cosine_sim = (cosine_sim > flag).float()      # ← 计算图在此断开
sen_loss = self.bce_loss(cosine_sim.float(), all_senlable.float())
loss += (1 / (iteration + 1)) * w * sen_loss
```

PyTorch 中比较运算返回**不可导的布尔张量**，`.float()` 只做 dtype 转换、不会恢复计算图。
实测 `binar.requires_grad == False`、`grad_fn is None`，单独反传直接抛
`RuntimeError: element 0 of tensors does not require grad`。因此
**∂L_CGL/∂θ ≡ 0**：该项只改变打印出来的 loss 数值，不改变任何参数的梯度。

影响范围：`all_senlable` / `all_senlable2` 由 `sen.csv`、`sen_more.csv` 读入后仅作为
`batch[8]`、`batch[9]` 透传，唯一的消费点就是上面的 `sen_loss`。也就是说，
**整条 Stage-1 LLM 伪标签链路对本实现的训练没有任何贡献**。

证据（Fold 1，其余配置完全相同）：

| 配置 | `--cgl_w` | 首轮损失 | F1 | 句内 | 跨句 | 最优轮次 |
|---|---|---|---|---|---|---|
| E1 原实现 | 6 | 8.7606 | 0.6110 | 0.7867 | 0.5223 | 26 |
| E0 去 CGL | 0 | 0.1026 | 0.6110 | 0.7867 | 0.5223 | 26 |

两行不是「四舍五入后相同」，而是 16 位有效数字逐位相同
（`F1 = 0.6110381077529566`，跨句 `F1 = 0.5222551923867566`）。
首轮损失相差 8.66，说明这一项确实被算进了 loss —— 它只是不产生梯度。

**修复**：新增 `--cgl_mode ste`，用直通估计器替换硬阈值

```
s̃ = 1[cos > τ] + cos − sg(cos)
```

前向数值与硬阈值完全相同（后两项相消），反向把梯度原样传回。默认值 `hard`
与原实现一致，不改变任何现有行为的数值。

**修复后的对照**（E3 vs E1，Fold 1，验证集 F1；受单卡算力限制 E3 跑到第 12 轮）：

| 轮次 | 1 | 3 | 6 | 9 | 12 |
|---|---|---|---|---|---|
| E1（CGL 无梯度） | 0.3127 | 0.4387 | 0.4313 | 0.4546 | 0.4720 |
| E3（STE，CGL 有梯度） | 0.1888 | 0.2265 | 0.3612 | 0.4242 | 0.4519 |

首轮损失 E3 为 9.1017、E1 为 8.7606（E0 仅 0.1026），证明前向一致、梯度已接通；
E3 在全部 12 轮里每一轮都落后（前 12 轮均值 0.3354 vs 0.4319，低 9.7 个点），
说明在 `w=6` 且伪标签近似"两事件所在句"时，CGL 的梯度对主任务是净负担。

> 补充：`(事件对数, 句子数, 768)` 的展开张量在 STE 模式下必须保留以便反传，
> 6 GB 卡上会触及显存换页（慢约 3 倍）。可替换为数学等价的
> `cosine_sim = event_pair @ sent_emb_n.squeeze(1).t()`（实测最大偏差 5.6e-8），
> 本仓库为与报告数字逐位对应仍保留原写法。

## 本仓库相对官方仓库的改动

官方仓库发布时缺失 4 类文件，其中 `utils.py` 被 `main.py` 直接 import，缺失即无法启动；
伪标签 CSV 缺失时代码会**静默回退为全零标签**（训练照常跑但学不到东西）。

| 文件 | 类型 | 内容 |
|---|---|---|
| `_repo/model/model.py` | 性能 | 5 处向量化改写，实测加速 12–17×，已验证数值等价 |
| `_repo/model/model.py` | 功能 | CGL 权重 `w=6` 硬编码 → `--cgl_w`（默认 6.0） |
| `_repo/model/model.py` | 修复 | 删除死代码 `true_sent`；修复 `sent_ground` 重复 append |
| `_repo/model/model.py` | 实验 | 新增 `--cgl_mode ste`（直通估计，默认 `hard` 与原实现一致） |
| `_repo/parameter.py` | 功能 | 新增 `--grad_ckpt`、`--cgl_w`、`--cgl_mode`、`--fold_start/--fold_end` |
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

消融（加 `--fold_start 1 --fold_end 1` 只跑 Fold 1）：

| 实验 | 关键参数 |
|---|---|
| E0 去 CGL | `--cgl_w 0` |
| E3 梯度修复 | `--cgl_mode ste` |

一键串行：`bash run_ablation.sh`（日志写入 `logs_ablation.txt`）。

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

# 下次会话续办清单（2026-11-12 00:35 保存现场）

## 当前进度
- 报告 `report/report.pdf`：32 页，0 错误 / 0 Overfull / 0 未定义引用，已含 CGL 缺陷完整章节。
- PPT `ppt/deck/deck.pptx`：11 页，新增 CGL 发现页，已全部 lint 通过并目视核对。
- 代码改动已提交 git（commit 047210a），**推送失败**（代理 502），本地 commit 完整。

## 核心发现（已完成，勿丢）
官方实现 `(cosine_sim > flag).float()` 切断计算图 → CGL 无梯度。
E0(w=0) 与 E1(w=6) Fold1 逐位相同（F1=0.6110381077529566，最优轮次均 26），
首轮损失 0.1026 vs 8.7606。整条 Stage-1 伪标签链路对训练零贡献。

## 消融实验状态
| 实验 | 状态 | 已得数据 |
|---|---|---|
| E0 (cgl_w=0) | 完成 Fold1，与 E1 逐位相同 | logs_E0_no_cgl.txt |
| E1 (cgl_w=6) | 5 折完成 | logs_E1_full.txt，F1=0.6493±0.0261 |
| E3 (STE 修复) | **被关机中断于第 12/30 轮** | logs_ablation.txt；第 12 轮 valid F1=0.4519（对照 E1 同轮 0.44），首轮 loss 9.1017 vs E1 8.7606 → 梯度确实生效 |
| E2 (cgl_w=3) | 未跑（排在 E3 后被杀） | — |
| E4 (单轮迭代) | 未跑（同上） | — |

E3 权重 `_repo/checkpoints/E3_ste_termgrad/fold1_best.pt`（best valid F1=0.4519）已保存。

## 待办
1. 决定 E3：重跑完（约 2 小时）`bash run_ablation.sh`，或用已有的 12 轮轨迹写成"梯度生效但收敛更慢"的部分结果。
2. 填报告：跑 `python tools/make_ablation_table.py --write`（解析日志自动填表），
   再把 `report/sections/06_experiment.tex` 的 `%%E3_BLOCK%%`、`%%E4_BLOCK%%` 替换成正文；
   若 E2/E4 不跑，删掉对应小节与附录表里的两行。
3. 重编译：`cd report && xelatex report.tex` 两遍（记得 `export MIKTEX_DISABLE_UPDATE_CHECK=1`）。
4. 推送 GitHub：`git -c http.proxy=http://127.0.0.1:49153 -c https.proxy=http://127.0.0.1:49153 push origin master`
   （若 502，换网络时段重试；分支是 master 不是 main）。
5. 打包：`bash make_package.sh` → 生成 `智能课设2026第4个项目_第六组_张旭+路文博.rar`。
   （WinRAR 已装到 `C:\Program Files\WinRAR\rar.exe`）
6. PPT 第 11 页若有数字变化（页数/结论），同步改后重新 upsert。

## 环境备忘
- 训练 venv：`D:\wangbclass\cogeci\.venv`（PyTorch 2.6.0+cu126）
- xelatex：`C:\Users\Lenovo\AppData\Local\Programs\MiKTeX\miktex\bin\x64\`（需 PATH + MIKTEX_DISABLE_UPDATE_CHECK=1）
- 渲染 PDF 用 `D:\wangbclass\cogeci\.venv\Scripts\python.exe` 的 fitz
- 代理：env 里有 `http_proxy=http://127.0.0.1:49153`

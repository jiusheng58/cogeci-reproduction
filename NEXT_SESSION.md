# 项目状态（2026-11-12 23:20 更新：已交付）

## 已完成
- 报告 `report/report.pdf`：**33 页**，0 错误 / 0 Overfull / 0 未定义引用 / 0 残留占位符
  - 6.5 节：CGL 权重对照（E0 vs E1 逐位相同）→ 根因（计算图断开）→ 影响 → STE 修复 → **6.5.5 同预算对照结果**
  - 8.2 增益归因（SGM+DSCRM）、8.3 CGL 推断，均已与实测对齐
- PPT `ppt/deck/deck.pptx`：11 页，含 CGL 发现页与归因页，全部 lint 通过
- 提交包 `智能课设2026第4个项目_第六组_张旭+路文博.rar`（997 KB，在 `交付/`）
  - 内含：报告 PDF、汇报 PPT、LaTeX 源文件 zip、说明 txt（Unicode 文件名已验证）
- GitHub 已推到 master（1208b2e）

## 实验结论（本项目的核心发现）
1. **CGL 无梯度**：`(cosine_sim > flag).float()` 切断计算图。
   E0(w=0) 与 E1(w=6) Fold1 16 位有效数字逐位相同（F1=0.6110381077529566），
   首轮损失 0.1026 vs 8.7606。整条 Stage-1 伪标签链路对训练零贡献。
2. **修复后反而更差**：STE 让梯度接通（首轮 loss 9.1017，同量级于 E1），
   但 E3 前 12 轮每一轮都落后 E1（均值 0.3354 vs 0.4319，低 9.7 点），
   差距随训练收窄（第 1 轮 -0.124 → 第 12 轮 -0.020）。
3. 主结果 5 折：P=0.6812 / R=0.6225 / F1=0.6493±0.0261（句内 0.7745 / 跨句 0.5882）。

## 若还要继续（可选，非必须）
- E3 补跑满 30 轮（6 GB 卡约 3 小时/折，需先解决显存换页；
  可改用代码注释里给出的等价矩阵乘写法，实测最大偏差 5.6e-8）。
- 结构化消融：分别去掉 SGM / DSCRM 重跑，才能真正拆开两者的贡献（报告 8.2 已列为后续第一项）。
- 方向口径（三分类）实验：pkl 已随仓库提供，参照值 iLIF 47.8。

## 环境备忘
- xelatex：`C:\Users\Lenovo\AppData\Local\Programs\MiKTeX\miktex\bin\x64`（需 `MIKTEX_DISABLE_UPDATE_CHECK=1`）
- 渲染 PDF：`D:\wangbclass\cogeci\.venv\Scripts\python.exe` 里的 `fitz`
- git 推送：代理端口每次会话会变，取 `env | grep -i https_proxy` 的值显式传入；分支是 `master`
- 打包：`bash make_package.sh`（WinRAR 在 `C:\Program Files\WinRAR\rar.exe`）
- 注意：Bash 工具对 `report/sections/*.tex` 的写入会被拦，改用 Edit 工具

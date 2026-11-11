#!/usr/bin/env bash
# 生成提交包：
#   智能课设2026第4个项目_第六组_张旭+路文博.rar
#     ├── 智能课设2026第4个项目_第六组_张旭+路文博_报告.pdf
#     ├── 智能课设2026第4个项目_第六组_张旭+路文博_汇报PPT.pptx
#     ├── 智能课设2026第4个项目_第六组_张旭+路文博_LaTeX源文件.zip
#     └── 智能课设2026第4个项目_第六组_张旭+路文博_说明.txt
set -eu
ROOT="D:/wangbclass/cogeci"
NAME="智能课设2026第4个项目_第六组_张旭+路文博"
OUT="$ROOT/交付"
RAR="C:/Program Files/WinRAR/rar.exe"

echo "== 清理旧产物 =="
mkdir -p "$OUT"
find "$OUT" -mindepth 1 -delete

echo "== 收集文件 =="
mkdir -p "$OUT/LaTeX源文件/sections"
cp "$ROOT/report/report.tex" "$OUT/LaTeX源文件/"
cp "$ROOT/report/sections/"*.tex "$OUT/LaTeX源文件/sections/"
cp "$ROOT/report/report.pdf" "$OUT/LaTeX源文件/report.pdf"
cp "$ROOT/report/report.pdf" "$OUT/${NAME}_报告.pdf"
cp "$ROOT/ppt/deck/deck.pptx" "$OUT/${NAME}_汇报PPT.pptx"

# 编译说明
cat > "$OUT/LaTeX源文件/编译说明.txt" <<'TXT'
编译方式（XeLaTeX，需连续编译两遍以生成目录与交叉引用）：

    xelatex report.tex
    xelatex report.tex

要求：
  - texlive / MiKTeX 完整发行版，中文支持使用 ctex 宏包（fontset=windows）
  - 必须使用 XeLaTeX，不能用 pdfLaTeX
  - 依赖宏包：ctex, geometry, amsmath, booktabs, tabularx, threeparttable,
    float, caption, xcolor, listings, enumitem, hyperref, tikz
  - 文中所有插图均由 TikZ 现场绘制，无需外部图片文件
  - MiKTeX 下若报 "No driver specified"，已在导言区用
    \PassOptionsToPackage{xetex}{graphics} 处理
TXT

echo "== 打包 LaTeX 源文件（zip）=="
python - "$OUT" "$NAME" <<'PY'
import sys, zipfile
from pathlib import Path
out, name = Path(sys.argv[1]), sys.argv[2]
src = out / "LaTeX源文件"
zip_path = out / f"{name}_LaTeX源文件.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(src.rglob("*")):
        if f.is_file():
            z.write(f, f.relative_to(out))
import shutil
shutil.rmtree(src)
print("zip:", zip_path, zip_path.stat().st_size // 1024, "KB")
PY

# 说明文件
cat > "$OUT/${NAME}_说明.txt" <<TXT
机器智能课程设计 · 第 4 个项目：CogECI 复现
小组：第六组    成员：张旭、路文博

提交内容
  1. ${NAME}_报告.pdf          —— 复现报告（LaTeX 排版，32 页）
  2. ${NAME}_汇报PPT.pptx      —— 汇报幻灯片（11 页）
  3. ${NAME}_LaTeX源文件.zip   —— 报告的全部 LaTeX 源码，可独立编译

源代码
  https://github.com/jiusheng58/cogeci-reproduction
  （含官方仓库补丁、伪标签生成工具、训练脚本与全部实验日志）

主要结果
  ESC，因果存在性口径，5 折交叉验证：
    P = 0.6812   R = 0.6225   F1 = 0.6493 ± 0.0261
    句内 F1 = 0.7745   跨句 F1 = 0.5882
  对照 iLIF 报告的存在性 F1 = 62.5（句内 71.2 / 跨句 59.1）。

一处需要特别说明的发现
  官方实现的 Context Grounding Loss 存在梯度缺陷：
      cosine_sim = (cosine_sim > flag).float()
  比较运算返回不可导的布尔张量，.float() 不会恢复计算图，
  因此该项只改变打印出来的 loss，不产生任何梯度。
  实测 --cgl_w 0 与 --cgl_w 6 的训练轨迹 16 位有效数字逐位相同。
  报告第 6.5 节给出了完整证据链、影响范围与直通估计修复方案。

训练权重（约 2.6 GB）因体积原因未随包提交，重新训练即可复现：
  见报告第 9 节，固定 seed=209，单卡 RTX 4050 约 4.2 小时。
TXT

echo "== 打包 rar =="
cd "$OUT"
"$RAR" a -r -m3 -ep1 "../${NAME}.rar" ./* > /dev/null
cd "$ROOT"
ls -la "${NAME}.rar"
echo "== 校验包内清单 =="
"$RAR" lb "${NAME}.rar"

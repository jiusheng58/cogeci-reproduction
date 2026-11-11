#!/usr/bin/env python
"""从训练日志中提取消融实验的 Fold 1 结果，重写 report/sections/06_experiment.tex
中的 %%ABL_BEGIN%% ... %%ABL_END%% 表格行。

用法：
    python tools/make_ablation_table.py            # 只打印解析结果
    python tools/make_ablation_table.py --write    # 同时写回 .tex
"""
import argparse
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "report" / "sections" / "06_experiment.tex"

LOG_E1 = ROOT / "logs_E1_full.txt"
LOG_E0 = ROOT / "logs_E0_no_cgl.txt"
LOG_ABL = ROOT / "logs_ablation.txt"

FOLD_RE = re.compile(
    r"Fold 1: R (?P<r>[0-9.]+) P (?P<p>[0-9.]+) F (?P<f>[0-9.]+) "
    r"intra\{'epoch': (?P<ie>\d+), 'p': (?P<ip>[0-9.]+), 'r': (?P<ir>[0-9.]+), 'f1': (?P<if1>[0-9.]+)\} "
    r"cross\{'epoch': (?P<ce>\d+), 'p': (?P<cp>[0-9.]+), 'r': (?P<cr>[0-9.]+), 'f1': (?P<cf>[0-9.]+)\}"
)
LOSS_RE = re.compile(r"Epoch1: Train Loss (?P<v>[0-9.]+)")


def _normalize(text: str) -> str:
    """进度条用 \\r 刷新，先按 \\r 与 \\n 全部打散。"""
    return text.replace("\r", "\n")


def parse_stream(text: str):
    """返回 (fold1 dict or None, first_epoch_loss or None)。取最后一次出现的 Fold 1。"""
    text = _normalize(text)
    folds = FOLD_RE.findall(text)
    loss = LOSS_RE.search(text)
    fold = None
    if folds:
        m = FOLD_RE.finditer(text)
        last = list(m)[-1]
        fold = {k: float(v) if k not in ("ie", "ce") else int(v) for k, v in last.groupdict().items()}
    return fold, (float(loss.group("v")) if loss else None)


def split_ablation(text: str):
    """把 logs_ablation.txt 按 '===== START <name>' 切段。"""
    text = _normalize(text)
    parts = re.split(r"=====\s*START\s+(\S+)[^\n]*\n", text)
    out = {}
    for i in range(1, len(parts), 2):
        out[parts[i]] = parts[i + 1]
    return out


def fmt(x, nd=4):
    return "---" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    runs = []  # (label, w, fold_dict, loss)

    if LOG_E1.exists():
        f, l = parse_stream(LOG_E1.read_text(encoding="utf-8", errors="ignore"))
        runs.append(("E1（原实现）", "6", f, l))
    if LOG_E0.exists():
        f, l = parse_stream(LOG_E0.read_text(encoding="utf-8", errors="ignore"))
        runs.append(("E0（去 CGL）", "0", f, l))

    if LOG_ABL.exists():
        segs = split_ablation(LOG_ABL.read_text(encoding="utf-8", errors="ignore"))
        for name, label, w in [
            ("E3_ste_termgrad", "E3（梯度修复）", "6"),
            ("E2_cgl_w3", "E2（权重减半）", "3"),
            ("E4_iter1", "E4（单轮迭代）", "9→1"),
        ]:
            if name in segs:
                f, l = parse_stream(segs[name])
                if f is not None:
                    runs.append((label, w, f, l))

    # 输出顺序：E1、E2、E0、E3、E4
    order = {"E1（原实现）": 0, "E2（权重减半）": 1, "E0（去 CGL）": 2,
             "E3（梯度修复）": 3, "E4（单轮迭代）": 4}
    runs.sort(key=lambda r: order.get(r[0], 9))

    rows, summary = [], []
    for label, w, f, l in runs:
        if f is None:
            summary.append(f"{label:14s} 无结果")
            continue
        rows.append(
            f"{label} & {w} & {fmt(f['p'])} & {fmt(f['r'])} & {fmt(f['f'])} & "
            f"{fmt(f['if1'])} & {fmt(f['cf'])} & {f['ie']} & {fmt(l)} \\\\"
        )
        summary.append(
            f"{label:14s} P={f['p']:.4f} R={f['r']:.4f} F1={f['f']:.4f} "
            f"intra={f['if1']:.4f} cross={f['cf']:.4f} best_epoch={f['ie']} first_loss={fmt(l)}"
        )

    print("\n".join(summary))
    print()
    print("\n".join(rows))

    if args.write and rows:
        tex = REPORT.read_text(encoding="utf-8")
        begin, end = "%%ABL_BEGIN%%", "%%ABL_END%%"
        i, j = tex.index(begin), tex.index(end)
        tex = tex[: i + len(begin)] + "\n" + "\n".join(rows) + "\n" + tex[j:]
        REPORT.write_text(tex, encoding="utf-8")
        print("\n已写回", REPORT)


if __name__ == "__main__":
    main()

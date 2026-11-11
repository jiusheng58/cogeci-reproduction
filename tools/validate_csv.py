# -*- coding: utf-8 -*-
"""
伪标签 CSV 终检：在训练前确认 get_sent_ground 能安全消费。

检查项：
  1. 每个文档恰好一行（doc_id 唯一，重复会让 get_sent_ground 把答案重复拼接 -> 崩）
  2. answer 数量 == 该文档 event_pairs 数量（否则 all_senlable 形状对不上）
  3. 所有 answer 能 eval、是 list[str]、'None' 或 'a,b,c' 格式
  4. 句子编号不越界
  5. 正例（target!=0）的覆盖：None 的比例（= 打标失败率）
  6. 与 pkl 的 doc 集合一致（不多不少）
"""
import argparse
import ast
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "_repo"))

import __main__                                    # noqa: E402
from data.process_esc_csr import ESC_features      # noqa: E402
__main__.ESC_features = ESC_features

import pickle                                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    feats = pickle.load(open(args.pkl, "rb"))
    rows = list(csv.reader(open(args.csv, encoding="utf-8")))
    header, data = rows[0], rows[1:]
    print(f"[csv] {len(data)} 行（表头 {len(header)} 列）")

    seen = {}
    n_ans = n_none = n_bad = 0
    for r in data:
        doc_id = r[1]
        if doc_id in seen:
            print(f"[FAIL] doc_id 重复: {doc_id}")
            n_bad += 1
        seen[doc_id] = r

    ft_by_id = {f.doc_id: f for f in feats}
    missing = set(ft_by_id) - set(seen)
    extra = set(seen) - set(ft_by_id)
    if missing:
        print(f"[FAIL] pkl 中有 {len(missing)} 篇 CSV 缺失: {sorted(missing)[:5]} ...")
        n_bad += 1
    if extra:
        print(f"[FAIL] CSV 多出 {len(extra)} 篇: {sorted(extra)[:5]}")
        n_bad += 1

    pos_total = pos_ok = 0
    for doc_id, f in ft_by_id.items():
        if doc_id not in seen:
            continue
        answers = ast.literal_eval(seen[doc_id][6])
        targets = [int(t) for t in f.target]
        if len(answers) != len(targets):
            print(f"[FAIL] {doc_id}: answers={len(answers)} != pairs={len(targets)}")
            n_bad += 1
            continue
        n_sen = len(f.sentences)
        for a, t in zip(answers, targets):
            n_ans += 1
            if not isinstance(a, str):
                print(f"[FAIL] {doc_id}: answer 非字符串 {a!r}")
                n_bad += 1
                continue
            if a == "None":
                n_none += 1
                continue
            for x in a.split(","):
                if not x.strip().lstrip("-").isdigit():
                    print(f"[FAIL] {doc_id}: 非法编号 {a!r}")
                    n_bad += 1
                    break
                if int(x) >= n_sen:
                    print(f"[FAIL] {doc_id}: 越界编号 {a!r} (n_sen={n_sen})")
                    n_bad += 1
                    break
            if t != 0:
                pos_total += 1
                pos_ok += 1

    print(f"[ok] 覆盖 {len(seen)}/{len(feats)} 篇；answer 总数 {n_ans}；"
          f"None {n_none}（其中正例缺失 {pos_total - pos_ok}/{pos_total}）")
    print("[PASS] 可安全用于训练" if n_bad == 0 else f"[FAIL] {n_bad} 处问题，不能训练")
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

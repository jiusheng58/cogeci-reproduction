# -*- coding: utf-8 -*-
"""
CogECI · Stage 1 的**离线替代方案**：不用任何 LLM，直接从 pkl 构造 grounded context 伪标签。

动机：LLM 免费额度已耗尽，但论文的 Stage 1 目标只是"给每个因果正例对找支撑它的句子集合"。
这件事可以退化为一个高置信的规则：

  strict  : 事件 a 所在句 ∪ 事件 b 所在句            （最小支撑集）
  loose   : [min(s1,s2), max(s1,s2)] 全部句 ∪ 前后各一句（区间 + 上下文窗口）

依据（可验证）：读 pkl 的 node_event 拿到的事件句编号是可靠的（用 event_span.py 反推过 span 验证），
且上一轮 27b 的 LLM 标注显示，模型给出的答案绝大多数正好就是这两个事件句（见一致性对照实验）。

输出格式与 data/process_esc_csr.py::get_sent_ground 完全兼容：
  id, doc_id, text, sentences, event_pair, node_event_id, answer(id)

用法：
  python tools/pseudo_from_rule.py --split train --variant strict --out_csv ...
"""
import argparse
import csv
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "_repo"))

import __main__                                    # noqa: E402
from data.process_esc_csr import ESC_features      # noqa: E402
__main__.ESC_features = ESC_features


def build_answers(ft, variant, cap=8, radius=1):
    """返回该文档每个 pair 的答案字符串列表（与 ft.event_pairs 同序）。

    variant="strict"（对应 model.py 里的 all_senlable，用于句内 intra 对）
        只取两个事件所在句。实测与 27b 的 LLM 答案句级 P=0.994 / F1=0.864，
        LLM 含两个事件句的比例 99.4%（intra）/ 96.5%（inter）。

    variant="loose"（对应 all_senlable2，用于句间 inter 对）
        取两事件句之间的完整区间；若区间过长（> cap）则退化为两事件句 ± radius 的局部窗口，
        避免出现"整篇都标 1"的退化标签（train 里有 20% 的 inter 对区间超过半篇）。
        dev 上与 LLM loose 答案 F1=0.746、R=0.745，是各候选规则中最好的。
    """
    sids = [int(ne[0]) for ne in ft.node_event]
    n_sent = len(ft.sentences)
    targets = [int(t) for t in ft.target]
    answers = []
    for pi, (a, b) in enumerate(ft.event_pairs):
        if targets[pi] == 0:
            answers.append("None")
            continue
        s1, s2 = sids[int(a)], sids[int(b)]
        lo, hi = min(s1, s2), max(s1, s2)
        if variant == "strict" or lo == hi:
            sel = {s1, s2}
        elif hi - lo + 1 <= cap:
            sel = set(range(lo, hi + 1))
        else:
            sel = {s1, s2}
            for x in (s1, s2):
                for d in range(1, radius + 1):
                    if x - d >= 0:
                        sel.add(x - d)
                    if x + d <= n_sent - 1:
                        sel.add(x + d)
        answers.append(",".join(str(x) for x in sorted(sel)))
    return answers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "dev"])
    ap.add_argument("--variant", required=True, choices=["strict", "loose"])
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--pkl", default=None)
    args = ap.parse_args()

    pkl = args.pkl or os.path.join(
        ROOT, "_repo/data/database/exist",
        f"EventStoryLine_bert_intra_and_inter_{args.split}_features.pkl")
    feats = pickle.load(open(pkl, "rb"))
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)

    n_pos = n_sel = 0
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "doc_id", "text", "sentences", "event_pair", "node_event_id", "answer(id)"])
        for d_i, ft in enumerate(feats):
            numbered = [f"{i}. {' '.join(s)}" for i, s in enumerate(ft.sentences)]
            answers = build_answers(ft, args.variant)
            for a in answers:
                if a != "None":
                    n_pos += 1
                    n_sel += len(a.split(","))
            w.writerow([d_i, ft.doc_id, ft.enc_text, numbered,
                        ft.event_pairs, ft.event_pairs, answers])

    print(f"[rule] split={args.split} variant={args.variant} docs={len(feats)} "
          f"positive={n_pos} avg_sent={n_sel / max(1, n_pos):.2f} -> {args.out_csv}")


if __name__ == "__main__":
    main()

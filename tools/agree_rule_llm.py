# -*- coding: utf-8 -*-
"""
一致性对照：规则伪标签 vs LLM（qwen3.8-27b）伪标签。

回答两个问题：
  Q1 规则法选出的句子，有多大比例是 LLM 也选的？（precision）
  Q2 LLM 选出的句子，有多大比例被规则法覆盖？（recall）
      —— 若 recall 很高，说明"LLM 的 grounded context 基本就是事件句及其邻域"，
         那么规则法可以作为一个合理的离线替代。

另外单独统计一条最有说服力的指标：
      「LLM 的答案是否完整包含两个事件所在句」，即 {s_ev1, s_ev2} ⊆ L 的比例。
      这个比例越高，说明规则法的假设越成立（不依赖我们选了哪套规则）。
"""
import argparse
import ast
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


def load_csv(path):
    rows = list(csv.reader(open(path, encoding="utf-8")))[1:]
    return {r[1]: ast.literal_eval(r[6]) for r in rows}


def to_set(a):
    if a == "None":
        return set()
    return {int(x) for x in a.split(",") if x.strip()}


def compare(pkl, llm_csv, rule_csv, label, verbose_n=3):
    feats = {f.doc_id: f for f in pickle.load(open(pkl, "rb"))}
    llm, rule = load_csv(llm_csv), load_csv(rule_csv)

    tp = fp = fn = 0          # micro（句子级）
    pair_n = 0
    f1s, jacs = [], []
    subset_of_llm = 0         # R ⊆ L
    llm_superset_event = 0    # {s1,s2} ⊆ L
    llm_empty = 0
    examples = []

    for doc_id, f in feats.items():
        if doc_id not in llm or doc_id not in rule:
            continue
        L_all, R_all = llm[doc_id], rule[doc_id]
        targets = [int(t) for t in f.target]
        sids = [int(ne[0]) for ne in f.node_event]
        for pi, t in enumerate(targets):
            if t == 0 or pi >= len(L_all) or pi >= len(R_all):
                continue
            L, R = to_set(L_all[pi]), to_set(R_all[pi])
            if not L:                     # LLM 没打出标（额度中断/解析失败）→ 跳过
                llm_empty += 1
                continue
            pair_n += 1
            inter = len(L & R)
            tp += inter
            fp += len(R - L)
            fn += len(L - R)
            f1 = 2 * inter / max(1e-9, len(L) + len(R))
            f1s.append(f1)
            jacs.append(inter / max(1e-9, len(L | R)))
            if R and R <= L:
                subset_of_llm += 1
            a, b = [int(x) for x in f.event_pairs[pi]]
            if {sids[a], sids[b]} <= L:
                llm_superset_event += 1
            elif len(examples) < verbose_n:
                examples.append((doc_id, pi, sorted(L), sorted(R)))

    P = tp / max(1, tp + fp)
    Rc = tp / max(1, tp + fn)
    F1 = 2 * P * Rc / max(1e-9, P + Rc)
    print(f"\n=== {label} ===")
    print(f"  可比 pair 数        : {pair_n}  (LLM 空答案跳过 {llm_empty})")
    print(f"  句级 micro  P/R/F1  : {P:.3f} / {Rc:.3f} / {F1:.3f}")
    print(f"  pair 级  F1 (macro) : {sum(f1s) / max(1, len(f1s)):.3f}")
    print(f"  pair 级 Jaccard     : {sum(jacs) / max(1, len(jacs)):.3f}")
    print(f"  规则集 ⊆ LLM 集     : {subset_of_llm / max(1, pair_n):.1%}")
    print(f"  LLM 含两个事件句    : {llm_superset_event / max(1, pair_n):.1%}   <-- 规则法的核心假设")
    for e in examples:
        print(f"    反例 doc={e[0]} pair#{e[1]} LLM={e[2]} rule={e[3]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--llm_dir", default=os.path.join(ROOT, "_repo/data/examples/llm"))
    ap.add_argument("--rule_dir", default=os.path.join(ROOT, "_repo/data/examples/llm/rule"))
    args = ap.parse_args()

    pkl = os.path.join(ROOT, "_repo/data/database/exist",
                       f"EventStoryLine_bert_intra_and_inter_{args.split}_features.pkl")
    pairs = [
        ("rule strict VS llm strict", "llama3help_{}_sen.csv".format(args.split), "{}_strict.csv".format(args.split)),
        ("rule loose  VS llm strict", "llama3help_{}_sen.csv".format(args.split), "{}_loose.csv".format(args.split)),
        ("rule loose  VS llm loose", "llama3help_{}_sen_more.csv".format(args.split), "{}_loose.csv".format(args.split)),
        ("rule strict VS llm loose", "llama3help_{}_sen_more.csv".format(args.split), "{}_strict.csv".format(args.split)),
    ]
    for label, llm_f, rule_f in pairs:
        lp = os.path.join(args.llm_dir, llm_f)
        rp = os.path.join(args.rule_dir, rule_f)
        if not (os.path.isfile(lp) and os.path.isfile(rp)):
            print(f"\n[skip] 缺文件 {llm_f} 或 {rule_f}")
            continue
        compare(pkl, lp, rp, f"{args.split.upper()} · {label}")


if __name__ == "__main__":
    main()

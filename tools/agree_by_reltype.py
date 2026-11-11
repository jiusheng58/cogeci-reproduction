# -*- coding: utf-8 -*-
"""
按 rel_type 拆分的一致性分析：为 intra / inter 两类 pair 分别选最合适的规则。

背景（据 model/model.py:199-211）：
  ground_type==1（句内 intra）  -> 用 all_senlable  （对应 llama3help_*_sen.csv）
  ground_type==0（句间 inter）  -> 用 all_senlable2 （对应 llama3help_*_sen_more.csv）
所以两份文件不是"宽/严二选一"，而是"同句对/跨句对"各自的监督信号。

对 inter 对，候选规则：
  A 只取两个事件句 {s1,s2}
  B 取两事件句之间的完整区间 [lo..hi]
  C 区间 + 首尾各扩一句
并与 LLM（27b）的 strict / loose 两份答案分别量 P/R/F1。
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
    return set() if a == "None" else {int(x) for x in a.split(",") if x.strip()}


def candidate(s1, s2, n_sent, kind):
    lo, hi = min(s1, s2), max(s1, s2)
    if kind == "A":
        return {s1, s2}
    if kind == "B":
        return set(range(lo, hi + 1))
    sel = set(range(lo, hi + 1))
    if lo - 1 >= 0:
        sel.add(lo - 1)
    if hi + 1 <= n_sent - 1:
        sel.add(hi + 1)
    return sel


def run(pkl, llm_strict, llm_loose, split):
    feats = {f.doc_id: f for f in pickle.load(open(pkl, "rb"))}
    S, L = load_csv(llm_strict), load_csv(llm_loose)

    print(f"\n########## {split.upper()} ##########")
    for group, gid in (("intra(同句)", 1), ("inter(跨句)", 0)):
        for ref_name, ref in (("llm-strict", S), ("llm-loose", L)):
            for kind in ("A", "B", "C"):
                tp = fp = fn = n = 0
                core = 0
                for doc_id, f in feats.items():
                    if doc_id not in ref:
                        continue
                    ans = ref[doc_id]
                    targets = [int(t) for t in f.target]
                    sids = [int(ne[0]) for ne in f.node_event]
                    n_sent = len(f.sentences)
                    for pi, t in enumerate(targets):
                        if t == 0 or pi >= len(ans):
                            continue
                        is_intra = (sids[int(f.event_pairs[pi][0])] == sids[int(f.event_pairs[pi][1])])
                        if (gid == 1) != is_intra:
                            continue
                        Lset = to_set(ans[pi])
                        if not Lset:
                            continue
                        a, b = [int(x) for x in f.event_pairs[pi]]
                        R = candidate(sids[a], sids[b], n_sent, kind)
                        n += 1
                        tp += len(Lset & R)
                        fp += len(R - Lset)
                        fn += len(Lset - R)
                        if {sids[a], sids[b]} <= Lset:
                            core += 1
                if n == 0:
                    continue
                P = tp / max(1, tp + fp)
                Rc = tp / max(1, tp + fn)
                F1 = 2 * P * Rc / max(1e-9, P + Rc)
                print(f"  {group:11s} vs {ref_name:9s} rule={kind}: n={n:5d}  "
                      f"P={P:.3f} R={Rc:.3f} F1={F1:.3f}  LLM含两事件句={core / n:.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    args = ap.parse_args()
    pkl = os.path.join(ROOT, "_repo/data/database/exist",
                       f"EventStoryLine_bert_intra_and_inter_{args.split}_features.pkl")
    d = os.path.join(ROOT, "_repo/data/examples/llm")
    run(pkl, os.path.join(d, f"llama3help_{args.split}_sen.csv"),
        os.path.join(d, f"llama3help_{args.split}_sen_more.csv"), args.split)


if __name__ == "__main__":
    main()

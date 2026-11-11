# -*- coding: utf-8 -*-
"""
打标前的小规模配置基准：同一批 prompt，比较
  1) 开/关思考（thinking）
  2) 串行 vs 并发
的延迟与输出质量，避免贸然跑全量 5478 次。
"""
import argparse
import json
import os
import pickle
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "_repo"))
sys.path.insert(0, HERE)

from label_sentences import (build_prompt, parse_sent_numbers, event_mention,      # noqa: E402
                             doc_sentence_texts, load_features, OpenAICompat,
                             SYSTEM)
import __main__                                     # noqa: E402
from data.process_esc_csr import ESC_features       # noqa: E402
__main__.ESC_features = ESC_features


def build_jobs(feats, n_pairs):
    jobs = []
    for ft in feats:
        sents = doc_sentence_texts(ft)
        numbered = [f"{i}. {s}" for i, s in enumerate(sents)]
        targets = [int(t) for t in ft.target]
        pairs = ft.event_pairs
        rels = [int(r) for r in ft.rel_type] if hasattr(ft, "rel_type") else [0] * len(pairs)
        for pi, (a, b) in enumerate(pairs):
            if targets[pi] != 0:
                a, b = int(a), int(b)
                e1, n1, s1 = event_mention(ft, a)
                e2, n2, s2 = event_mention(ft, b)
                jobs.append((ft.doc_id, (a, b), e1, e2, rels[pi],
                             build_prompt(numbered, e1, e2, s1, s2, n1, n2,
                                          rels[pi], "strict"), len(sents)))
            if len(jobs) >= n_pairs:
                return jobs
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=os.path.join(
        ROOT, "_repo/data/database/exist/EventStoryLine_bert_intra_and_inter_dev_features.pkl"))
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--models", default=os.environ.get("OPENAI_MODEL", "qwen3.8-27b"))
    ap.add_argument("--concurrency", default="1")
    ap.add_argument("--think", default="0")   # 0=关思考, 1=开思考
    args = ap.parse_args()

    feats = load_features(args.pkl)
    jobs = build_jobs(feats, args.n)
    print(f"[bench] {len(jobs)} prompts, models={args.models}, "
          f"concurrency={args.concurrency}, think={args.think}\n")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL")

    _tls = threading.local()

    def init(model, no_think):
        _tls.llm = OpenAICompat(model, base_url, api_key, no_think, 120)

    for model in args.models.split(","):
        for think in args.think.split(","):
            no_think = (think.strip() == "0")
            for conc in [int(c) for c in args.concurrency.split(",")]:
                t0 = time.time()

                def work(j):
                    doc_id, pr, e1, e2, rel, msg, ns = j
                    t = time.time()
                    try:
                        out, reason = _tls.llm.generate(msg, 0.0)
                        err = ""
                    except Exception as ex:
                        out, reason, err = "", "", f"{type(ex).__name__}: {str(ex)[:120]}"
                    return doc_id, pr, e1, e2, rel, out, parse_sent_numbers(out, ns), err, time.time() - t

                results = []
                if conc <= 1:
                    init(model, no_think)
                    results = [work(j) for j in jobs]
                else:
                    with ThreadPoolExecutor(max_workers=conc,
                                            initializer=init, initargs=(model, no_think)) as ex:
                        results = list(ex.map(work, jobs))

                el = time.time() - t0
                ok = sum(1 for r in results if r[6] is not None)
                nsel = sum(len(r[6].split(",")) for r in results if r[6])
                print(f"--- model={model} think={int(not no_think)} conc={conc} ---")
                print(f"    total={el:.1f}s  per_call={el / len(jobs):.1f}s  "
                      f"parsed={ok}/{len(jobs)}  avg_sent={nsel / max(1, ok):.2f}")
                for r in results[:4]:
                    print(f"    [{r[0]} {r[1]}] e1={r[2]!r} e2={r[3]!r} rel={r[4]} "
                          f"-> {r[6]!r}  ({r[8]:.1f}s) raw={r[5][:60]!r}")
                    if r[7]:
                        print(f"        ERR {r[7]}")
                print()


if __name__ == "__main__":
    main()

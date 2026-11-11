# -*- coding: utf-8 -*-
"""
显存与耗时实测：6GB 显卡到底能不能跑得动 CogECI 的一个 epoch。

做法：从 train pkl 里挑出 最短 / 随机 / 最长 三类文档，逐篇跑一次完整
"前向 + 反向 + optimizer.step"，记录峰值显存和耗时，再外推一个 epoch 的总耗时。

用法：
  D:/wangbclass/cogeci/.venv/Scripts/python.exe tools/bench_vram.py
  D:/wangbclass/cogeci/.venv/Scripts/python.exe tools/bench_vram.py --n_long 5 --n_rand 5
"""
import argparse
import os
import pickle
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "_repo")
sys.path.insert(0, REPO)

import torch
from torch.utils.data import DataLoader, SequentialSampler

import __main__
from data.process_esc_csr import ESC_features
__main__.ESC_features = ESC_features

from parameter import parse_args  # noqa: E402
from model import bertCSRModel  # noqa: E402
from transformers import BertTokenizer, AdamW  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--n_short", type=int, default=3)
    ap.add_argument("--n_rand", type=int, default=5)
    ap.add_argument("--n_long", type=int, default=5)
    ap.add_argument("--grad_ckpt", action="store_true", help="开启 BERT 梯度检查点，压显存")
    ap.add_argument("--max_iteration", type=int, default=0, help="覆盖迭代轮数（0=用默认）")
    args = ap.parse_args()

    sys.argv = [sys.argv[0]]
    margs = parse_args()
    margs.cache_path = os.path.join(REPO, "data", "database", "exist")
    margs.model_name = os.path.join(REPO, "model", "bert-base-uncased")
    margs.no_of_classes = 2
    margs.device = torch.device("cuda")

    pkl = os.path.join(margs.cache_path,
                       f"EventStoryLine_bert_intra_and_inter_{args.split}_features.pkl")
    with open(pkl, "rb") as f:
        feats = pickle.load(f)
    print(f"{args.split}: {len(feats)} docs")

    # 用"送进 BERT 的 token 总数"当长度指标
    def ntok(ft):
        t = ft.enc_input_ids
        return int(t.numel()) if hasattr(t, "numel") else sum(len(x) for x in t)

    f0 = feats[0]
    print("enc_input_ids shape:", tuple(f0.enc_input_ids.shape),
          "| enc_mask_ids shape:", tuple(f0.enc_mask_ids.shape))

    order = sorted(range(len(feats)), key=lambda i: ntok(feats[i]))
    idx = (order[:args.n_short]
           + random.sample(order, args.n_rand)
           + order[-args.n_long:])
    idx = list(dict.fromkeys(idx))

    tok = BertTokenizer.from_pretrained(margs.model_name)
    model = bertCSRModel(margs).to(margs.device)
    if args.grad_ckpt:
        model.pretrained_model.gradient_checkpointing_enable()
        print("[grad_ckpt] BERT 梯度检查点已开启")
    if args.max_iteration:
        model.max_iteration = args.max_iteration + 1
        print(f"[iter] max_iteration 覆盖为 {args.max_iteration}")
    optimizer = AdamW(model.parameters(), lr=margs.lr)

    from data.process_esc_csr import ESC_processor
    processor = ESC_processor(margs, tok, printlog=lambda *a, **k: None)

    results = []
    oom = 0
    print("\n" + "=" * 78)
    print(f"{'doc_id':<12}{'tokens':>8}{'pairs':>7}{'sent':>6}{'peak MB':>10}{'sec':>8}")
    print("-" * 78)

    for i in idx:
        ft = feats[i]
        ds = processor.convert_features_to_dataset([ft])
        loader = DataLoader(ds, batch_size=1, sampler=SequentialSampler(ds),
                            collate_fn=ds.collate_fn)
        batch = next(iter(loader))
        inputs = {
            "enc_input_ids": batch[0].to(margs.device), "enc_mask_ids": batch[1].to(margs.device),
            "node_event": batch[2], "t1_pos": batch[3], "t2_pos": batch[4], "target": batch[5],
            "rel_type": batch[6], "event_pairs": batch[7], "all_senlable": batch[8],
            "all_senlable2": batch[9], "type": "train", "doc_id": batch[10], "tokenizer": tok,
        }
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model.train()
        optimizer.zero_grad()
        t0 = time.time()
        try:
            loss, opt = model(**inputs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), margs.max_grad_norm)
            optimizer.step()
            err = ""
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                oom += 1
                err = "OOM"
            else:
                raise
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1024 ** 2
        n_sent = len(ft.sentences)
        print(f"{str(ft.doc_id):<12}{ntok(ft):>8}{len(ft.target):>7}{n_sent:>6}"
              f"{peak:>10.0f}{dt:>8.2f}  {err}")
        if not err:
            results.append((ntok(ft), dt, peak))
        torch.cuda.empty_cache()

    print("=" * 78)
    if results:
        # 按 token 数加权外推整份数据集耗时
        tot_tok = sum(ntok(f) for f in feats)
        bench_tok = sum(r[0] for r in results)
        bench_t = sum(r[1] for r in results)
        est = bench_t / bench_tok * tot_tok
        peak_max = max(r[2] for r in results)
        print(f"样本 {len(results)} 篇，OOM {oom} 篇")
        print(f"峰值显存最高 {peak_max:.0f} MB  (显卡共 "
              f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB)")
        print(f"外推 1 个 epoch ≈ {est/60:.1f} 分钟")
        print(f"外推 5-fold × 30 epoch ≈ {est*30*5/3600:.1f} 小时")
    else:
        print("全部 OOM —— 6GB 装不下，需要降内存方案")


if __name__ == "__main__":
    main()

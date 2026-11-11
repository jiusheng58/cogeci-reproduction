# -*- coding: utf-8 -*-
"""
性能补丁的数值等价性验证：model/model.py（向量化版） vs model/_orig_model.py（原版备份）。

同一文档、同一随机种子下分别实例化两个模型，比较：
  1. train 模式的 loss 与 prediction（含反向后梯度范数）
  2. eval 模式的 loss 与 prediction（走 rate=w 的另一条代码路径）
  3. get_graphedge_index 在随机 prediction 下的输出（train/eval 两种模式）
  4. 单文档前向+反向耗时对比（速度收益）
"""
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "_repo")
sys.path.insert(0, REPO)
os.chdir(REPO)                      # 相对路径 ./data 必须以 _repo 为工作目录

import torch                                        # noqa: E402
import __main__                                     # noqa: E402
from data.process_esc_csr import ESC_features, ESC_processor  # noqa: E402
__main__.ESC_features = ESC_features

from parameter import parse_args as _pa             # noqa: E402
sys.path.insert(0, HERE)
_argv = sys.argv
sys.argv = [sys.argv[0]]
model_args = _pa()
sys.argv = _argv
model_args.device = torch.device("cuda")
model_args.no_of_classes = 2

from model import bertCSRModel as NewModel          # noqa: E402
from model._orig_model import bertCSRModel as OrigModel  # noqa: E402


def build_model(cls, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return cls(model_args).to(model_args.device)


def grab_batch(n=0):
    with open("data/database/exist/EventStoryLine_bert_intra_and_inter_dev_features.pkl", "rb") as f:
        feats = pickle.load(f)
    from data.process_esc_csr import ESC_dataset
    ds = ESC_dataset(feats)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=1, collate_fn=ds.collate_fn)
    it = iter(loader)
    for _ in range(n + 1):
        batch = next(it)
    return batch


def fwd(model, batch, training):
    if training:
        model.train()
    else:
        model.eval()
    inputs = {
        "enc_input_ids": batch[0].to(model_args.device),
        "enc_mask_ids": batch[1].to(model_args.device),
        "node_event": batch[2], "t1_pos": batch[3], "t2_pos": batch[4],
        "target": batch[5], "rel_type": batch[6], "event_pairs": batch[7],
        "all_senlable": batch[8].to(model_args.device),
        "all_senlable2": batch[9].to(model_args.device),
        "type": "train", "doc_id": batch[10], "tokenizer": None,
    }
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        t0 = time.time()
        loss, pred = model(**inputs)
        dt = time.time() - t0
        if training:
            model.zero_grad()
            loss.backward()
            gn = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        else:
            gn = 0.0
    return loss.item(), pred.detach(), gn, dt


def main():
    torch.backends.cudnn.deterministic = True
    print("[equiv] 对比 3 篇文档（含最长文档），train + eval 两种模式\n")
    all_ok = True
    for n_doc, doc_id_hint in [(0, "首篇"), (2, "第3篇"), (9, "第10篇")]:
        batch = grab_batch(n_doc)
        print(f"--- doc {batch[10]} pairs={len(batch[5][0])} sents={batch[0].shape[0]} ---")
        for mode in ("train", "eval"):
            seed = 3407
            m1 = build_model(OrigModel, seed)
            l1, p1, g1, t1 = fwd(m1, batch, mode == "train")
            m2 = build_model(NewModel, seed)
            l2, p2, g2, t2 = fwd(m2, batch, mode == "train")
            dl = abs(l1 - l2)
            dp = (p1 - p2).abs().max().item()
            dg = abs(g1 - g2) / max(g1, 1e-9)
            ok = dl < 1e-4 and dp < 1e-4 and dg < 1e-3
            all_ok &= ok
            print(f"  [{mode:5s}] loss {l1:.6f} vs {l2:.6f} (Δ={dl:.2e})  "
                  f"predΔ={dp:.2e}  grad范数Δrel={dg:.2e}  "
                  f"耗时 {t1:.2f}s -> {t2:.2f}s ({t1/max(t2,1e-9):.1f}x)  {'OK' if ok else 'FAIL'}")
            del m1, m2
            torch.cuda.empty_cache()
        print()

    # 单独对比 get_graphedge_index
    print("--- get_graphedge_index 直接对比（随机 prediction） ---")
    N = 500
    torch.manual_seed(0)
    pred = torch.randn(N, 2).to(model_args.device)
    pairs = [[i % 37, (i * 7) % 37] for i in range(N)]
    rel_type = [[i % 2 for i in range(N)]]
    m1 = build_model(OrigModel, 1)
    m2 = build_model(NewModel, 1)
    for mode in ("train", "eval"):
        m1.train(mode == "train"); m2.train(mode == "train")
        with torch.no_grad():
            g1 = m1.get_graphedge_index(pred, pairs, rel_type)
            g2 = m2.get_graphedge_index(pred, pairs, rel_type)
        ok = set(g1.keys()) == set(g2.keys())
        if ok:
            for k in g1:
                ok &= bool(torch.equal(g1[k].long(), g2[k].long()))
        all_ok &= ok
        n_e = {k: v.shape for k, v in g1.items()}
        print(f"  [{mode:5s}] {'OK ' if ok else 'FAIL'} edges={n_e}")
    print("\n" + "=" * 60)
    print("EQUIVALENCE", "PASSED" if all_ok else "FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

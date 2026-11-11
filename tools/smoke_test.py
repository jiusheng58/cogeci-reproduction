# -*- coding: utf-8 -*-
"""
CogECI 冒烟测试：验证 依赖 / 数据 / 伪标签 / 前向传播 是否都能跑通。

用法（用 D 盘 venv）：
  D:/wangbclass/cogeci/.venv/Scripts/python.exe tools/smoke_test.py
  D:/wangbclass/cogeci/.venv/Scripts/python.exe tools/smoke_test.py --with_labels   # 校验伪标签真的加载了

检查项：
  1. torch / CUDA / transformers / torch_geometric 可用
  2. ESC pkl 能加载，字段齐全
  3. bert-base-uncased 能加载
  4. 伪标签 all_senlable 形状 == [事件对数, 句子数]，且不全零
  5. 完整前向 + 反向能跑通
"""
import argparse
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "_repo")
sys.path.insert(0, REPO)

import torch  # noqa: E402

# pkl 里的类引用是 __main__.ESC_features（main.py 顶部重复定义过这个类），
# 工具脚本必须先把它注册回 __main__，否则 pickle.load 报 Can't get attribute。
import __main__  # noqa: E402
from data.process_esc_csr import ESC_features, ESC_processor  # noqa: E402
__main__.ESC_features = ESC_features

ok = True


def check(name, cond, extra=""):
    global ok
    flag = "PASS" if cond else "FAIL"
    if not cond:
        ok = False
    print(f"[{flag}] {name} {extra}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_path", default=os.path.join(REPO, "data", "database", "exist"))
    ap.add_argument("--model_name", default=os.path.join(REPO, "model", "bert-base-uncased"))
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n_docs", type=int, default=2)
    ap.add_argument("--with_labels", action="store_true")
    args = ap.parse_args()

    print("=" * 60)
    print(f"torch {torch.__version__} | cuda available = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
    check("CUDA 可用", torch.cuda.is_available())

    import transformers
    print(f"transformers {transformers.__version__}")
    from transformers import AdamW, BertTokenizer, BertModel  # noqa: F401
    check("transformers 导出 AdamW / BertTokenizer / BertModel", True)

    import torch_geometric
    print(f"torch_geometric {torch_geometric.__version__}")
    from torch_geometric.nn.conv import GraphConv, MessagePassing  # noqa: F401
    check("torch_geometric 核心 API", True)

    # ---- 数据 ----
    pkl = os.path.join(args.cache_path, f"EventStoryLine_bert_intra_and_inter_{args.split}_features.pkl")
    check("pkl 存在", os.path.exists(pkl), pkl)
    with open(pkl, "rb") as f:
        feats = pickle.load(f)
    need = {"doc_id", "topic_id", "sentences", "node_event", "t1_pos", "t2_pos",
            "target", "rel_type", "event_pairs", "enc_input_ids", "enc_mask_ids"}
    check("feature 字段齐全", need.issubset(set(vars(feats[0]).keys())))

    # ---- tokenizer / bert ----
    tok = BertTokenizer.from_pretrained(args.model_name)
    check("bert-base-uncased 加载", True, args.model_name)

    # ---- dataset + collate ----
    from parameter import parse_args as _pa
    _argv = sys.argv
    sys.argv = [sys.argv[0]]          # parse_args 会读命令行，先清掉
    model_args = _pa()
    sys.argv = _argv
    model_args.cache_path = args.cache_path
    model_args.model_name = args.model_name
    model_args.no_of_classes = 2
    model_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = ESC_processor(model_args, tok, printlog=print)
    _, dataset = processor.generate_dataloader(args.split)
    from torch.utils.data import DataLoader, SequentialSampler
    loader = DataLoader(dataset, batch_size=1, sampler=SequentialSampler(dataset),
                        collate_fn=dataset.collate_fn)

    n = 0
    for batch in loader:
        enc_input_ids, enc_mask_ids, node_event, t1_pos, t2_pos, target, rel_type, \
            event_pairs, all_senlable, all_senlable2, doc_id = batch
        n += 1
        n_pairs = len(target[0])
        n_sent = all_senlable.shape[1]
        print(f"\n--- doc {doc_id} : pairs={n_pairs} sentences={n_sent} ---")
        check("all_senlable 形状 == [事件对数, 句子数]",
              tuple(all_senlable.shape) == (n_pairs, n_sent),
              f"{tuple(all_senlable.shape)} vs {(n_pairs, n_sent)}")
        if args.with_labels:
            nz = int((all_senlable != 0).sum())
            check("伪标签非全零（说明 CSV 真的加载了）", nz > 0, f"非零元素={nz}")
        else:
            nz = int((all_senlable != 0).sum())
            if nz == 0:
                print("       [提示] 全零 = 当前是'无伪标签回退'，跑打标脚本生成 CSV 后再验一次")
        if n >= args.n_docs:
            break

    # ---- 前向 + 反向 ----
    from model import bertCSRModel
    model = bertCSRModel(model_args).to(model_args.device)
    batch = next(iter(loader))
    inputs = {
        "enc_input_ids": batch[0].to(model_args.device),
        "enc_mask_ids": batch[1].to(model_args.device),
        "node_event": batch[2], "t1_pos": batch[3], "t2_pos": batch[4],
        "target": batch[5], "rel_type": batch[6], "event_pairs": batch[7],
        "all_senlable": batch[8], "all_senlable2": batch[9],
        "type": "train", "doc_id": batch[10], "tokenizer": tok,
    }
    loss, opt = model(**inputs)
    check("前向通过", True, f"loss={loss.item():.4f} prediction={tuple(opt.shape)}")
    loss.backward()
    check("反向通过", True)

    print("\n" + "=" * 60)
    print("SMOKE TEST", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

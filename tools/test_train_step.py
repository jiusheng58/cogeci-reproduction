# -*- coding: utf-8 -*-
"""
验证 main.py 真实训练链路里最容易炸的两件事：
  1. utils.setup_seed 里的 torch.use_deterministic_algorithms(True) 在 CUDA 上是否报错
  2. 模型权重能否 save / 重载（课程"模型能重载"要求）
外加跑一个完整的 optimizer.step()，确认训练循环可行。
"""
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "_repo")
sys.path.insert(0, REPO)

import torch
from torch.utils.data import DataLoader, SequentialSampler

import __main__
from data.process_esc_csr import ESC_features, ESC_processor
__main__.ESC_features = ESC_features

from utils import setup_seed, compute_f1, transfor3to2, calculate  # noqa: E402
from parameter import parse_args  # noqa: E402
from model import bertCSRModel  # noqa: E402
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup  # noqa: E402

sys.argv = [sys.argv[0]]
args = parse_args()
args.cache_path = os.path.join(REPO, "data", "database", "exist")
args.model_name = os.path.join(REPO, "model", "bert-base-uncased")
args.no_of_classes = 2
args.epoch = 1
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("setup_seed(209) ...")
try:
    setup_seed(209)
    print("  [PASS] setup_seed 通过（确定性算法未报错）")
except Exception as e:
    print(f"  [FAIL] setup_seed 报错: {type(e).__name__}: {e}")
    print("  -> 需把 utils.py 的 torch.use_deterministic_algorithms(True, warn_only=False) 改成 warn_only=True")

tok = BertTokenizer.from_pretrained(args.model_name)
processor = ESC_processor(args, tok, printlog=lambda *a, **k: None)
_, dev_dataset = processor.generate_dataloader("dev")
loader = DataLoader(dev_dataset, batch_size=1, sampler=SequentialSampler(dev_dataset),
                    collate_fn=dev_dataset.collate_fn)

model = bertCSRModel(args).to(args.device)
no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
opt_groups = [
    {"params": [p for n, p in model.pretrained_model.named_parameters() if not any(nd in n for nd in no_decay)],
     "weight_decay": args.weight_decay, "lr": args.t_lr},
    {"params": [p for n, p in model.pretrained_model.named_parameters() if any(nd in n for nd in no_decay)],
     "weight_decay": 0.0, "lr": args.t_lr},
    {"params": [p for n, p in model.mlp.named_parameters() if not any(nd in n for nd in no_decay)],
     "weight_decay": args.weight_decay},
    {"params": [p for n, p in model.CGE.named_parameters() if not any(nd in n for nd in no_decay)],
     "weight_decay": args.weight_decay},
]
optimizer = AdamW(opt_groups, lr=args.lr, eps=args.adam_epsilon)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1, num_training_steps=10)

batch = next(iter(loader))
inputs = {
    "enc_input_ids": batch[0].to(args.device), "enc_mask_ids": batch[1].to(args.device),
    "node_event": batch[2], "t1_pos": batch[3], "t2_pos": batch[4], "target": batch[5],
    "rel_type": batch[6], "event_pairs": batch[7], "all_senlable": batch[8],
    "all_senlable2": batch[9], "type": "train", "doc_id": batch[10], "tokenizer": tok,
}
model.train()
loss, opt = model(**inputs)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
optimizer.step()
scheduler.step()
model.zero_grad()
print(f"  [PASS] 训练步通过 loss={loss.item():.4f} prediction={tuple(opt.shape)}")

gold_ = [t for bt in inputs["target"] for t in bt]
pred_ = list(torch.argmax(opt, -1).cpu().numpy())
gold, pred = transfor3to2(gold_, pred_)
p, r, f1 = compute_f1(gold, pred, __import__("logging").getLogger("x"))
print(f"  [PASS] 指标可计算 P={p:.4f} R={r:.4f} F1={f1:.4f}（伪标签全零时 F1 为 0 属正常）")

ckpt = os.path.join(HERE, "..", "checkpoints", "smoke.pt")
os.makedirs(os.path.dirname(ckpt), exist_ok=True)
torch.save(model.state_dict(), ckpt)
model2 = bertCSRModel(args).to(args.device)
model2.load_state_dict(torch.load(ckpt, map_location=args.device))
print(f"  [PASS] 权重保存/重载通过 -> {os.path.abspath(ckpt)}")

print("\nTRAIN-STEP TEST PASSED")

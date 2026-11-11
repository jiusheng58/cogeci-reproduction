# -*- coding: utf-8 -*-
"""
按真实提示词估算 Stage 1 打标的 token 消耗与费用。

关键点：原实现的提示里 **每个事件对都要带上整篇文档的编号句子列表**，
所以一篇文档有 N 个正例，这篇文档的文本就要被重复送 N 次 —— 这才是主要成本。

用法：
  D:/wangbclass/cogeci/.venv/Scripts/python.exe tools/estimate_tokens.py
"""
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "_repo")
sys.path.insert(0, REPO)

import __main__
from data.process_esc_csr import ESC_features
__main__.ESC_features = ESC_features

sys.path.insert(0, HERE)
from label_sentences import build_prompt, event_mention, doc_sentence_texts  # noqa: E402

# 英文学术/新闻文本的经验换算：约 4 个字符 = 1 个 token
CHARS_PER_TOKEN = 4.0


def main():
    total_calls = 0
    total_in_tok = 0.0
    per_split = {}

    for split in ["train", "dev"]:
        pkl = os.path.join(REPO, "data", "database", "exist",
                           f"EventStoryLine_bert_intra_and_inter_{split}_features.pkl")
        with open(pkl, "rb") as f:
            feats = pickle.load(f)

        calls = 0
        in_tok = 0.0
        doc_rows = []
        for ft in feats:
            sents = doc_sentence_texts(ft)
            numbered = [f"{i}. {s}" for i, s in enumerate(sents)]
            targets = [int(t) for t in ft.target]
            n_pos = sum(1 for t in targets if t != 0)
            if n_pos == 0:
                continue
            # 取该文档第一个正例，构造一次完整提示词来量长度
            pairs = ft.event_pairs
            pi = next(i for i, t in enumerate(targets) if t != 0)
            a, b = int(pairs[pi][0]), int(pairs[pi][1])
            e1 = event_mention(ft, a)
            e2 = event_mention(ft, b)
            msg = build_prompt(numbered, e1, e2, e1, e2, "strict")
            chars = len(msg[0]["content"]) + len(msg[1]["content"])
            tok = chars / CHARS_PER_TOKEN
            # 该文档的重复系数
            doc_tok = tok * n_pos
            calls += n_pos
            in_tok += doc_tok
            doc_rows.append((ft.doc_id, len(sents), n_pos, tok, doc_tok))

        per_split[split] = (len(feats), calls, in_tok, doc_rows)
        total_calls += calls
        total_in_tok += in_tok

        print(f"\n===== {split} =====")
        print(f"文档 {len(feats)} 篇 | 正例事件对 {calls} 次调用")
        print(f"输入 token ≈ {in_tok/1e6:.2f} M")
        top = sorted(doc_rows, key=lambda r: -r[4])[:5]
        print("最耗 token 的 5 篇：")
        for d, ns, np_, t, dt in top:
            print(f"  {d:<10} {ns:>3} 句 × {np_:>4} 个正例  单次 {t:>6.0f} tok  小计 {dt/1000:>7.1f} k tok")

    # 输出 token：每次回答形如 "[1,2,3]"，保守按 15 token 计
    out_tok = total_calls * 15

    print("\n" + "=" * 60)
    print(f"单份（strict 或 loose 之一）")
    print(f"  调用次数   {total_calls:,}")
    print(f"  输入 token ≈ {total_in_tok/1e6:.2f} M")
    print(f"  输出 token ≈ {out_tok/1000:.0f} k")
    print(f"\n两份（intra 用 strict、inter 用 loose，共 2 份）")
    print(f"  输入 token ≈ {total_in_tok*2/1e6:.2f} M")
    print(f"  输出 token ≈ {out_tok*2/1000:.0f} k")

    print("\n--- 费用粗估（按各家官网当前价，仅供量级参考）---")
    tin, tout = total_in_tok * 2 / 1e6, out_tok * 2 / 1e6
    plans = [
        ("gpt-4o-mini", 0.15, 0.60, 7.2),
        ("DeepSeek-V3 类", 0.28, 0.42, 7.2),
        ("通义 qwen-plus 类", 0.55, 1.65, 7.2),
    ]
    for name, pin, pout, fx in plans:
        usd = tin * pin + tout * pout
        print(f"  {name:<20} ≈ ${usd:5.2f}  ≈ ¥{usd*fx:5.1f}")

    print("\n提示：若嫌贵，可把同一篇文档的多个正例合并进一次调用（文档文本只送一次），")
    print("      输入 token 可降到约 1/5 ~ 1/10，但这偏离论文逐对提问的设定，需在报告写明。")


if __name__ == "__main__":
    main()

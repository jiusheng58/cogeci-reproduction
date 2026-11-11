# -*- coding: utf-8 -*-
"""
从 node_event 的 BERT 位置反推事件的真实词级 span。

原理（读 _ilif/features_pkl_process/process_event.py 得出）：
  构造句子时往 token 列表里插了两个 '[CLS]' 标记：
    s.insert(start, '[CLS]'); s.insert(start + n, '[CLS]')
  再做 BERT 编码（add_special_tokens=True），然后 delete_tokens(..., index=1, 101) 删掉
  第二个 101（第一个插入的标记）得到 v1，再删一次得到 v2。

  设 L(i) = 前 i 个词的 BERT 子词总数，则
    v1 = 1 + L(start)          # 1 是句首自带的 [CLS]
    v2 = 1 + L(end)            # 删掉 marker1 后 marker2 左移一位
  于是 start = L^{-1}(v1 - 1)，end = L^{-1}(v2 - 1)。

  直接用 sentences[sid][v1:v2] 切片是错的——v1/v2 是子词下标，
  只要事件里有 magnitude/Indonesia 这种会被切成多个子词的词就会整体错位，
  切出来是 ','、'in'、'2004' 这种噪声。
"""
import os
import sys
from functools import lru_cache

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")


@lru_cache(maxsize=1)
def _tok():
    from transformers import BertTokenizer
    return BertTokenizer.from_pretrained(os.path.join(ROOT, "_repo", "model", "bert-base-uncased"))


def _cum_lens(tokens):
    """L[i] = 前 i 个词的子词总数，长度 len(tokens)+1"""
    tk = _tok()
    L = [0]
    for t in tokens:
        L.append(L[-1] + len(tk.tokenize(t)))
    return L


def invert(L, target):
    """找到 L[i] == target 的 i；找不到就取最接近的（保证 start<=end）。"""
    if target <= 0:
        return 0
    if target >= L[-1]:
        return len(L) - 1
    for i, v in enumerate(L):
        if v == target:
            return i
    # 落在某个词的中间（子词边界），取该词起点
    for i in range(len(L) - 1):
        if L[i] < target < L[i + 1]:
            return i
    return len(L) - 1


def event_span_text(sent_tokens, v1, v2):
    """返回事件的词级文本（已 strip）。"""
    L = _cum_lens(sent_tokens)
    s = invert(L, v1 - 1)
    e = invert(L, v2 - 1)
    if e <= s:
        e = min(len(sent_tokens), s + 1)
    return " ".join(sent_tokens[s:e]).strip(), (s, e)


if __name__ == "__main__":
    import pickle
    sys.path.insert(0, os.path.join(ROOT, "_repo"))
    import __main__
    from data.process_esc_csr import ESC_features
    __main__.ESC_features = ESC_features

    pkl = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, "_repo/data/database/exist/EventStoryLine_bert_intra_and_inter_dev_features.pkl")
    feats = pickle.load(open(pkl, "rb"))
    for ft in feats[:3]:
        print(f"===== doc {ft.doc_id} =====")
        for i, ne in enumerate(ft.node_event[:22]):
            sid = int(ne[0])
            v1, v2 = int(ne[1][0]), int(ne[1][1])
            toks = ft.sentences[sid]
            old = " ".join(toks[max(0, v1):min(len(toks), v2)])
            new, (s, e) = event_span_text(toks, v1, v2)
            print(f"{i:2d} s{sid:2d} v=({v1},{v2})  旧={old!r:32s} 新[{s}:{e}]={new!r}")

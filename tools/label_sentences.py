# -*- coding: utf-8 -*-
"""
CogECI · Stage 1：用 LLM 给每个"因果正例事件对"抽取 grounded context 句子编号。

为什么不用仓库自带的 llm/run_llm_processing.py：
  它用 `for j in range(n-1): for z in range(j+1, n)` 的字典序遍历去索引 `target`，
  但 pkl 里的 `event_pairs` 并不是字典序（例如 dev 首篇是 [[0,1],[2,0],[3,0],[0,4],...]），
  两者顺序不一致会让伪标签整体错位。本脚本直接按 pkl 的 event_pairs / target 遍历，顺序天然对齐。

事件提及文本：
  node_event 里的 [v1,v2] 是 BERT 子词位置，不能直接切 sentences[sid][v1:v2]，
  要用 tools/event_span.py 反推词级 span（否则抽出来是 ','、'in'、'2004' 这种噪声）。

输出 CSV 列（与 data/process_esc_csr.py::get_sent_ground 完全兼容）：
  id, doc_id, text, sentences, event_pair, node_event_id, answer(id)
  - row[1] = doc_id        -> get_sent_ground 按它匹配
  - row[6] = answer(id)    -> eval 后是 list[str]，元素形如 "1,2" 或 "None"

同时把每次调用的提示词原文与模型原始响应写入 --raw_jsonl，满足课程"留有提示与原始响应"的要求。

用法：
  python tools/label_sentences.py \
      --pkl _repo/data/database/exist/EventStoryLine_bert_intra_and_inter_train_features.pkl \
      --out_csv _repo/data/examples/llm/mine_train_sen.csv \
      --raw_jsonl _repo/data/examples/llm/raw_train_sen.jsonl \
      --variant strict --concurrency 8 --no_think
"""
import argparse
import csv
import json
import os
import pickle
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "_repo"))
sys.path.insert(0, HERE)

from event_span import event_span_text   # noqa: E402


# ----------------------------- .env -----------------------------
def load_dotenv(path):
    """极简 .env 解析：KEY=VALUE，跳过 # 注释与空行。不覆盖已有环境变量。"""
    if not os.path.isfile(path):
        return {}
    got = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if not k:
                continue
            got[k] = v
            os.environ.setdefault(k, v)
    return got


load_dotenv(os.path.join(ROOT, "_repo", ".env"))

# pkl 里的类引用是 __main__.ESC_features，必须先注册回 __main__ 才能 pickle.load
import __main__                                    # noqa: E402
from data.process_esc_csr import ESC_features      # noqa: E402
__main__.ESC_features = ESC_features


# ----------------------------- LLM provider -----------------------------
class OpenAICompat(object):
    def __init__(self, model, base_url, api_key, no_think=False, timeout=120, max_tokens=2048):
        from openai import OpenAI
        kw = {"api_key": api_key, "timeout": timeout, "max_retries": 0}
        if base_url:
            kw["base_url"] = base_url
        self.client = OpenAI(**kw)
        self.model = model
        self.no_think = no_think
        self.max_tokens = max_tokens

    def generate(self, messages, temperature):
        msgs = messages
        kw = dict(model=self.model, temperature=temperature, max_tokens=self.max_tokens)
        low = self.model.lower()
        if self.no_think:
            if "deepseek" in low:
                # DeepSeek 系：/no_think 后缀，extra_body 不生效
                msgs = [dict(m) for m in messages]
                msgs[-1] = {**msgs[-1], "content": msgs[-1]["content"] + " /no_think"}
            elif "qwen" in low:
                # Qwen3 系：兼容模式关思考
                kw["extra_body"] = {"enable_thinking": False,
                                    "chat_template_kwargs": {"enable_thinking": False}}
        r = self.client.chat.completions.create(messages=msgs, **kw)
        ch = r.choices[0]
        if ch.finish_reason == "length":
            # 思考把 max_tokens 吃光、答案被截断 -> 当作失败重试
            raise RuntimeError("truncated: finish_reason=length")
        m = ch.message
        txt = m.content or ""
        return txt, (getattr(m, "reasoning_content", None) or "")


SYSTEM = "You are an expert with few words and will only answer questions accurately"


def build_prompt(numbered, e1, e2, es1, es2, n1, n2, rel_type, variant):
    """尽量贴合 _repo/llm/run_llm_processing.py::_prompt_select_sentences 的原文。

    与官方的唯一差别：官方的 es1/es2 来自另一个 LLM 步骤（event_extension 生成的 SVO 短句），
    这里用"事件所在整句原文"代替（省掉近万次额外调用，报告中会注明该简化）。
    """
    if rel_type == 0:
        head = (f"knowing that event: {e1} and event: {e2} in the {n1} sentence are causally related, ")
        tail = (f"( they may be clustered in the {n1} sentence, "
                f"but this does not preclude the need for multiple sentences.) ")
    else:
        head = (f"knowing that event: {e1} and event: {e2} in the {n1} and {n2} sentences "
                f"are causally related, ")
        tail = (f"( they may be clustered in the {n1} and {n2} sentences, "
                f"but this does not preclude the need for others sentences.) ")
    user = (f"The input document is labeled with an ordinal number before each sentence, e.g. (0,1...) , "
            + head
            + f"{e1} means [{es1}].{e2} means[{es2}]"
            + f"identify which sentences from the document articulate the causal relationship "
              f"between events {e1} and {e2}. "
            + tail)
    if variant == "loose":
        user += ("Include every sentence that provides background, intermediate reasoning or coreference "
                 "needed to understand the causal link, not only the sentences containing the two events.")
    user += ('Answer Present the sentence numbers as a list, e.g., "[1,2,...]".don\'t output other content. '
             f"Document :{list(numbered)}")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse_sent_numbers(text, n_sent):
    """
    从模型输出里抽出句子编号。返回规范化字符串（"1,3,7"）或 None。

    规范化很重要：下游 get_sent_ground 用 `[int(x) for x in item.split(",")]`，
    任何非数字字符（"sentence 3"、"1 and 3"）都会让训练直接崩掉。
    这里统一：取最后一个方括号内的整数 -> 去重 -> 排序 -> 丢弃越界编号。
    """
    if not text:
        return None
    s, e = text.rfind("["), text.rfind("]")
    if s == -1 or e == -1 or e < s:
        return None
    nums, seen = [], set()
    for m in re.finditer(r"-?\d+", text[s + 1:e]):
        v = int(m.group())
        if v < 0 or v >= n_sent:      # 越界（含幻觉编号）直接丢弃
            continue
        if v not in seen:
            seen.add(v)
            nums.append(v)
    if not nums:
        return None
    return ",".join(str(x) for x in sorted(nums))


# ----------------------------- data helpers -----------------------------
def load_features(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


_span_cache = {}
_span_lock = threading.Lock()


def event_mention(ft, ev_idx):
    """返回 (提及文本, 所在句编号, 所在句原文)。带缓存，避免重复分词。"""
    key = (id(ft), ev_idx)
    with _span_lock:
        if key in _span_cache:
            return _span_cache[key]
    ne = ft.node_event[ev_idx]
    sid = int(ne[0])
    toks = ft.sentences[sid]
    text, _ = event_span_text(toks, int(ne[1][0]), int(ne[1][1]))
    if not text:
        text = " ".join(toks[:3]).strip() or f"event{ev_idx}"
    val = (text, sid, " ".join(toks).strip())
    with _span_lock:
        _span_cache[key] = val
    return val


def doc_sentence_texts(ft):
    return [" ".join(s) for s in ft.sentences]


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--raw_jsonl", default=None)
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    ap.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL"))
    ap.add_argument("--api_key_env", default="OPENAI_API_KEY")
    ap.add_argument("--variant", default="strict", choices=["strict", "loose"])
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 篇文档（试点用），0=全部")
    ap.add_argument("--start", type=int, default=0, help="从第 N 篇文档开始（断点续跑用）")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_retries", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=1, help="并发请求数")
    ap.add_argument("--no_think", action="store_true", help="关闭 Qwen3 思考模式（省 token、快很多）")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--dry_run", action="store_true", help="只统计调用量，不真的调 API")
    args = ap.parse_args()

    feats = load_features(args.pkl)
    if args.limit:
        feats = feats[args.start:args.start + args.limit]
    else:
        feats = feats[args.start:]

    total_pairs = sum(len(f.target) for f in feats)
    total_pos = sum(1 for f in feats for t in f.target if int(t) != 0)
    print(f"[stats] docs={len(feats)} pairs={total_pairs} positive={total_pos} "
          f"(API calls needed = {total_pos})")

    if args.dry_run:
        return

    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise RuntimeError(f"missing env {args.api_key_env}（检查 _repo/.env 是否写对）")
    print(f"[config] model={args.model} base_url={args.base_url} variant={args.variant} "
          f"temp={args.temperature} concurrency={args.concurrency} no_think={args.no_think}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    raw_f = open(args.raw_jsonl, "a", encoding="utf-8") if args.raw_jsonl else None
    raw_lock = threading.Lock()

    # 断点续跑：--start>0 且 CSV 已存在时改为追加，避免覆盖已完成的文档
    resume = args.start > 0 and os.path.exists(args.out_csv)
    csv_mode = "a" if resume else "w"

    stats = {"call": 0, "fail": 0, "none": 0, "sel": 0, "pos": 0, "tok": 0}
    stat_lock = threading.Lock()
    t0 = time.time()

    with open(args.out_csv, csv_mode, newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not resume:
            w.writerow(["id", "doc_id", "text", "sentences", "event_pair", "node_event_id", "answer(id)"])

        for d_i, ft in enumerate(feats):
            sents = doc_sentence_texts(ft)
            numbered = [f"{i}. {s}" for i, s in enumerate(sents)]
            targets = [int(t) for t in ft.target]
            pairs = ft.event_pairs
            rels = [int(r) for r in ft.rel_type] if hasattr(ft, "rel_type") else [0] * len(pairs)

            jobs = []          # (pi, a, b)
            for pi, (a, b) in enumerate(pairs):
                if targets[pi] != 0:
                    jobs.append((pi, int(a), int(b)))

            def work(job):
                pi, a, b = job
                e1, n1, s1 = event_mention(ft, a)
                e2, n2, s2 = event_mention(ft, b)
                msg = build_prompt(numbered, e1, e2, s1, s2, n1, n2, rels[pi], args.variant)
                out, reason = "", ""
                llm = _tls.llm
                for attempt in range(args.max_retries):
                    try:
                        out, reason = llm.generate(msg, args.temperature)
                        break
                    except Exception as ex:
                        msg_err = str(ex)
                        if "insufficient_quota" in msg_err or "Free quota exhausted" in msg_err:
                            # 额度耗尽是永久性错误，重试无意义，直接中止整个任务
                            print(f"[FATAL] API 免费额度耗尽，中止。已完成调用可用 --start 断点续跑：{msg_err[:200]}")
                            raise SystemExit(2)
                        with stat_lock:
                            stats["fail"] += 1
                        print(f"[warn] doc={ft.doc_id} pair=({a},{b}) try{attempt + 1} "
                              f"{type(ex).__name__}: {msg_err[:160]}")
                        time.sleep(2.0 * (attempt + 1))
                ans = parse_sent_numbers(out, len(sents))
                if ans is None:
                    with stat_lock:
                        stats["none"] += 1
                    ans = "None"
                else:
                    with stat_lock:
                        stats["sel"] += len(ans.split(","))
                        stats["pos"] += 1
                with stat_lock:
                    stats["call"] += 1
                if raw_f:
                    with raw_lock:
                        raw_f.write(json.dumps({
                            "doc_id": ft.doc_id, "pair": [a, b], "variant": args.variant,
                            "model": args.model, "prompt": msg, "raw_response": out,
                            "reasoning": reason, "parsed": ans, "n_sent": len(sents),
                        }, ensure_ascii=False) + "\n")
                        raw_f.flush()
                return pi, ans

            _tls = threading.local()
            results = {}
            if args.concurrency <= 1:
                _tls.llm = OpenAICompat(args.model, args.base_url, api_key,
                                        args.no_think, args.timeout)
                for job in jobs:
                    pi, ans = work(job)
                    results[pi] = ans
            else:
                def init():
                    _tls.llm = OpenAICompat(args.model, args.base_url, api_key,
                                            args.no_think, args.timeout)
                with ThreadPoolExecutor(max_workers=args.concurrency,
                                        initializer=init) as ex:
                    for pi, ans in ex.map(work, jobs):
                        results[pi] = ans

            answers = ["None"] * len(pairs)
            for pi, ans in results.items():
                answers[pi] = ans
            for pi, (a, b) in enumerate(pairs):
                if targets[pi] == 0:
                    answers[pi] = "None"

            w.writerow([d_i, ft.doc_id, ft.enc_text, numbered, pairs, pairs, answers])
            f.flush()
            el = time.time() - t0
            print(f"[done] {d_i + 1}/{len(feats)} doc={ft.doc_id} pairs={len(pairs)} "
                  f"calls={stats['call']} fail={stats['fail']} unparsable={stats['none']} "
                  f"avg_sent={stats['sel'] / max(1, stats['pos']):.2f} "
                  f"elapsed={el:.0f}s eta={el / (d_i + 1) * (len(feats) - d_i - 1):.0f}s")

    if raw_f:
        raw_f.close()
    print(f"[summary] calls={stats['call']} api_fail={stats['fail']} unparsable={stats['none']} "
          f"avg_selected_sent={stats['sel'] / max(1, stats['pos']):.2f} "
          f"elapsed={time.time() - t0:.0f}s -> {args.out_csv}")


if __name__ == "__main__":
    main()

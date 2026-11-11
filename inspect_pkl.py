import pickle, sys, collections


class _Stub(object):
    def __init__(self, *a, **kw):
        self.__dict__.update(kw)


class _U(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            return type(name, (object,), {"__init__": lambda self, *a, **kw: self.__dict__.update(kw)})


for split in ["train", "dev"]:
    p = f"D:/wangbclass/cogeci/_ilif/data/database/EventStoryLine_bert_intra_and_inter_{split}_features.pkl"
    with open(p, "rb") as f:
        feats = _U(f).load()
    print(f"\n===== {split}: {len(feats)} docs =====")
    f0 = feats[0]
    print("fields:", sorted(vars(f0).keys()))
    n_pairs = 0
    tg = collections.Counter()
    rt = collections.Counter()
    for ft in feats:
        t = list(getattr(ft, "target", []))
        r = list(getattr(ft, "rel_type", []))
        tg.update(t)
        rt.update(r)
        n_pairs += len(t)
    print("total event pairs:", n_pairs)
    print("target dist (0/1/2 = NONE/CAUSE/EFFECT):", dict(sorted(tg.items())))
    print("rel_type dist (0=intra 1=inter):", dict(sorted(rt.items())))
    print("positive pairs (target!=0):", sum(v for k, v in tg.items() if k != 0))
    sents = getattr(f0, "sentences", None)
    print("sample sentences type:", type(sents), "len:", len(sents) if sents is not None else None)
    print("sample node_event[0:3]:", list(getattr(f0, "node_event", []))[:3])
    print("sample event_pairs[0:3]:", list(getattr(f0, "event_pairs", []))[:3])
    print("topic ids (first/last):", getattr(feats[0], "topic_id", None), getattr(feats[-1], "topic_id", None))
    et = getattr(f0, "enc_text", None)
    print("enc_text type:", type(et), "len:", len(et) if et is not None else None)

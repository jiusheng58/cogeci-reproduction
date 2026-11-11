import pickle


class _U(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            return type(name, (object,), {"__init__": lambda self, *a, **kw: self.__dict__.update(kw)})


p = "D:/wangbclass/cogeci/_ilif/data/database/exist/EventStoryLine_bert_intra_and_inter_dev_features.pkl"
with open(p, "rb") as f:
    feats = _U(f).load()

ft = feats[0]
print("doc_id      :", ft.doc_id, "| topic_id:", ft.topic_id)
print("sentences   :", type(ft.sentences), len(ft.sentences))
print("  [0]:", repr(ft.sentences[0])[:120])
print("enc_tokens  :", type(ft.enc_tokens), len(ft.enc_tokens))
print("  [0]:", type(ft.enc_tokens[0]), repr(ft.enc_tokens[0])[:150])
print("enc_text    :", repr(ft.enc_text)[:200])
print("node_event  :", len(ft.node_event))
ne = ft.node_event[0]
print("  node_event[0] =", int(ne[0]), [int(x) for x in ne[1]])
ne2 = ft.node_event[1]
print("  node_event[1] =", int(ne2[0]), [int(x) for x in ne2[1]])
print("event_pairs :", len(ft.event_pairs), ft.event_pairs[:5])
print("t1_pos      :", len(ft.t1_pos), [list(map(int, t)) for t in ft.t1_pos[:3]])
print("t2_pos      :", len(ft.t2_pos), [list(map(int, t)) for t in ft.t2_pos[:3]])
print("target      :", len(ft.target), list(map(int, ft.target[:10])))
print("rel_type    :", len(ft.rel_type), list(map(int, ft.rel_type[:10])))

# try reconstructing event mention text
sent0 = ft.enc_tokens[0]
s, e = int(ne[1][0]), int(ne[1][1])
print("\nreconstruct event0 from enc_tokens[0][%d:%d] ->" % (s, e), sent0[s:e])

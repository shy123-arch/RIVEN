#!/usr/bin/env python3
"""Generate VAL with the 6-quantizer ViMo-free H1 (base masked + residual)."""
import argparse, json, os, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h1_train as HT
import cond_adapters as CA
from ordered_adapter import OrderedAdapter
import models.mask_transformer.transformer_memo_cross as TMC
from models.vq.model import RVQVAE

torch = HT.torch; dev = HT.dev; DATA = HT.DATA; CACHE = HT.CACHE


class TCPerceiver(nn.Module):
    def __init__(self, d=512, n_slots=16):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.q = nn.Parameter(torch.randn(n_slots, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, 8, batch_first=True)
        self.ln_mem = nn.LayerNorm(d)
        self.glob = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))

    def forward(self, g, m):
        h = self.proj(m)
        q = self.q[None].expand(h.shape[0], -1, -1)
        mem, _ = self.attn(q, h, h)
        return self.glob(g), self.ln_mem(mem + q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lengths", default="gt", choices=["gt", "pred"])
    ap.add_argument("--lengths-file", default="", help="npz with ids,lengths for predicted duration")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--cond", default="video", choices=["video", "text"])
    ap.add_argument("--cond-source", default="qwen", choices=["qwen", "tcclip", "fusion"])
    ap.add_argument("--adapter-type", default="perceiver",
                    choices=["perceiver", "ordered", "raw", "two_stream", "prompt_ordered",
                             "perceiver2", "perceiver4", "flat6m", "idq", "role"])
    ap.add_argument("--role-swap", action="store_true",
                    help="role-swap control: prefix from visual tokens, memory from prompt tokens")
    ap.add_argument("--role-inverse", action="store_true",
                    help="inverse role swap: pooled AEM prefix, expanded RIV memory")
    ap.add_argument("--m-source", default="visual", choices=["visual", "both"])
    ap.add_argument("--g-layer", type=int, default=36)
    ap.add_argument("--m-layers", default="18,27")
    ap.add_argument("--v-layer", type=int, default=27)
    ap.add_argument("--p-layer", type=int, default=18)
    ap.add_argument("--plan-slots", type=int, default=4)
    ap.add_argument("--role-mode", default="both", choices=["both", "visual", "plan"])
    ap.add_argument("--role-no-type", action="store_true")
    ap.add_argument("--role-no-time", action="store_true")
    ap.add_argument("--n-visual", type=int, default=16)
    ap.add_argument("--role-fields", default="none", choices=["none", "gt", "self"])
    ap.add_argument("--riv-pool", default="", choices=["", "mean", "attn", "last"])
    ap.add_argument("--n-enc-layers", type=int, default=1)
    ap.add_argument("--tc-adapter", action="store_true")
    ap.add_argument("--visual-cache", default=os.environ.get("RIVEN_VISUALFEAT", str(HT.CACHE / "visualfeat")))
    ap.add_argument("--visual-name", default="visual_trainval")
    ap.add_argument("--fusion-gate", action="store_true")
    ap.add_argument("--tc-val", default=os.environ.get("RIVEN_TC_VAL", ""))
    ap.add_argument("--slots", type=int, default=16)
    ap.add_argument("--split", default="val", choices=["val", "train", "test"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--res-scale", type=float, default=5.0)
    ap.add_argument("--cfg-mem", action="store_true")
    ap.add_argument("--intent-mode", default="", choices=["", "cross", "direct", "refresh"],
                    help="empty = read from checkpoint args")
    ap.add_argument("--cfg-mode", default="both", choices=["both", "intent", "evidence"],
                    help="which channel the CFG unconditional branch removes")
    ap.add_argument("--cond-scale", type=float, default=4.0)
    ap.add_argument("--timesteps", type=int, default=18)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--topk", type=float, default=0.9)
    ap.add_argument("--repr", default="textfeat", choices=["textfeat", "frame"])
    ap.add_argument("--repr-name", default="sft")
    ap.add_argument("--corpus", default="", help="override textfeat corpus name")
    ap.add_argument("--repr-cache", default=os.environ.get("RIVEN_TEXTFEAT", str(HT.CACHE / "textfeat")))
    ap.add_argument("--use-layers", default="36")
    ap.add_argument("--retr-file", default="")
    ap.add_argument("--nce-on-m", action="store_true")
    ap.add_argument("--save-codes", action="store_true")
    ap.add_argument("--ablate-g", default="none", choices=["none", "zero", "shuffle", "mean"])
    ap.add_argument("--ablate-m", default="none", choices=["none", "zero", "shuffle"])
    ap.add_argument("--null-branch", default="full", choices=["full", "r0", "e0", "both"],
                    help="learned-null inference branch: full (r,e), r0 (r0,e), e0 (r,e0), both (r0,e0)")
    ap.add_argument("--null-branch-base", default="", choices=["", "full", "r0", "e0", "both"],
                    help="override --null-branch for the base (mask) stage only")
    ap.add_argument("--null-branch-res", default="", choices=["", "full", "r0", "e0", "both"],
                    help="override --null-branch for the residual stage only")
    ap.add_argument("--tc-val-res", default="",
                    help="optional npz (same ids as --tc-val) used only by res.generate")
    ap.add_argument("--keys-file", default="", help="optional newline/JSON list of ids to generate")
    ap.add_argument("--save-base-codes", action="store_true", help="also store base-stage codes")
    ap.add_argument("--time-json", default="", help="optional path for per-stage generation timings")
    ap.add_argument("--riv-prefix", default="", choices=["", "cross", "residual", "direct"],
                    help="empty = read from checkpoint args; cross/residual/direct override")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    st = torch.load(args.ckpt, map_location="cpu")
    intent_mode = args.intent_mode or st.get("args", {}).get("intent_mode", "cross")
    drop_mode = st.get("args", {}).get("drop_mode", "joint")
    vq_ck = torch.load(st["vq"], map_location="cpu")
    mean, std = vq_ck["mean"], vq_ck["std"]; nb = vq_ck["nb_code"]; Q = vq_ck["num_quantizers"]
    a = SimpleNamespace(num_quantizers=Q, shared_codebook=False, quantize_dropout_prob=0.0,
                        quantize_dropout_cutoff_index=0, nb_code=nb, code_dim=512, mu=0.99)
    vq = RVQVAE(a, 263, nb, 512, 512, 2, 2, 512, 3, 3, "relu", None).to(dev)
    vq.load_state_dict(vq_ck["vq_model"]); vq.eval()
    opt = SimpleNamespace(num_tokens=nb, num_quantizers=Q, device=dev)
    if args.cfg_mem:
        import cfg_patch; cfg_patch.apply(TMC)
    import intent_patch; intent_patch.apply(TMC)
    if drop_mode == "independent":
        intent_patch.patch_independent_dropout(TMC)
    elif drop_mode == "g_only":
        intent_patch.patch_g_only_dropout(TMC)
    elif drop_mode == "learned_null":
        intent_patch.patch_learned_null(TMC)
    print(json.dumps({"intent_mode": intent_mode, "drop_mode": drop_mode}), flush=True)
    trans = TMC.MaskTransformer(code_dim=512, cond_mode="video", latent_dim=384, ff_size=1024,
                                num_layers=6, num_heads=6, dropout=0.2, clip_dim=512,
                                cond_drop_prob=0.1, opt=opt).to(dev)
    res = TMC.ResidualTransformer(code_dim=512, cond_mode="video", latent_dim=384, ff_size=1024,
                                  num_layers=6, num_heads=6, dropout=0.2, clip_dim=512,
                                  shared_codebook=False, opt=opt).to(dev)
    layers = tuple(int(x) for x in args.use_layers.split(","))
    if args.cond_source in ("tcclip", "fusion"):
        ad = None
    elif args.adapter_type == "raw":
        ad = CA.RawLinear(use_layers=layers).to(dev)
    elif args.adapter_type == "two_stream":
        ad = CA.TwoStream(g_layer=args.g_layer,
                          m_layers=tuple(int(x) for x in args.m_layers.split(",")),
                          n_slots=args.slots, n_enc_layers=args.n_enc_layers).to(dev)
    elif args.adapter_type == "prompt_ordered":
        ad = CA.OrderedPrompt(use_layers=layers, n_slots=args.slots,
                              n_enc_layers=args.n_enc_layers).to(dev)
    elif args.adapter_type == "perceiver2":
        ad = CA.Perceiver2(use_layers=layers, n_slots=args.slots).to(dev)
    elif args.adapter_type == "perceiver4":
        ad = CA.Perceiver4(use_layers=layers, n_slots=args.slots, n_layers=3).to(dev)
    elif args.adapter_type == "flat6m":
        ad = CA.Flat6M(g_layer=args.g_layer, v_layer=args.v_layer, p_layer=args.p_layer,
                       n_slots=args.slots, n_plan=args.plan_slots,
                       n_visual=args.n_visual).to(dev)
    elif args.adapter_type == "idq":
        ad = CA.IDQ(use_layers=layers, n_slots=args.slots).to(dev)
    elif args.adapter_type == "role":
        _riv_pool = args.riv_pool or st.get("args", {}).get("riv_pool", "mean")
        ad = CA.RoleAdapter(g_layer=args.g_layer, v_layer=args.v_layer, p_layer=args.p_layer,
                            n_slots=args.slots, n_plan=args.plan_slots,
                            use_visual=args.role_mode in ("both", "visual"),
                            use_plan=args.role_mode in ("both", "plan"),
                            use_type=not args.role_no_type, use_time=not args.role_no_time,
                            n_visual=args.n_visual, use_fields=args.role_fields != "none",
                            riv_pool=_riv_pool, swap=args.role_swap, inverse=args.role_inverse).to(dev)
    elif args.adapter_type == "ordered":
        ad = OrderedAdapter(use_layers=layers, n_slots=args.slots, m_source=args.m_source).to(dev)
    else:
        ad = HT.Adapter(use_layers=layers, n_slots=args.slots).to(dev)
    vmeta_vis = HT.load_visual(args.visual_name, args.visual_cache) \
        if args.adapter_type in ("ordered", "role", "perceiver4", "flat6m") else None
    tc_ad = TCPerceiver().to(dev) if args.cond_source == "tcclip" and args.tc_adapter else None
    qad = HT.VisualAdapter(use_layers=tuple(int(x) for x in args.use_layers.split(",")), n_slots=args.slots).to(dev) if args.cond_source == "fusion" else None
    fusion_gate = torch.nn.Parameter(torch.zeros((), device=dev)) if args.fusion_gate and args.cond_source == "fusion" else None
    fusion_ln = torch.nn.LayerNorm(512).to(dev) if fusion_gate is not None else None
    vmeta = HT.load_visual(args.visual_name, args.visual_cache) if args.cond_source == "fusion" else None
    proj = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 512)).to(dev)
    proj_m = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 512)).to(dev) if args.nce_on_m else None
    intent_patch.configure(trans, intent_mode, 0.0)
    intent_patch.configure(res, intent_mode, 0.0)
    riv_prefix = args.riv_prefix or st.get("args", {}).get("riv_prefix", "cross")
    trans.riv_prefix = res.riv_prefix = riv_prefix
    if drop_mode == "learned_null":
        branch_base = args.null_branch_base or args.null_branch
        branch_res = args.null_branch_res or args.null_branch
        intent_patch.configure_learned_null(trans)
        intent_patch.configure_learned_null(res)
        intent_patch.set_null_branch(trans, branch_base)
        intent_patch.set_null_branch(res, branch_res)
    elif args.null_branch_base or args.null_branch_res:
        print(json.dumps({"warning": "stage null flags ignored: drop_mode is not learned_null"}), flush=True)
    trans.cfg_mode = res.cfg_mode = args.cfg_mode
    trans.load_state_dict(st["trans"]); res.load_state_dict(st["res"])
    if ad is not None:
        ad.load_state_dict(st["adapter"], strict=False)
    if tc_ad is not None:
        tc_ad.load_state_dict(st["tc_adapter"])
    if qad is not None:
        qad.load_state_dict(st["q_adapter"])
    if fusion_gate is not None:
        fusion_gate.data.copy_(st["fusion_gate"])
    if fusion_ln is not None:
        fusion_ln.load_state_dict(st["fusion_ln"])
    if args.retr_file and "proj" in st:
        proj.load_state_dict(st["proj"]); proj.eval()
    if proj_m is not None:
        proj_m.load_state_dict(st["proj_m"]); proj_m.eval()
    trans.eval(); res.eval()
    if ad is not None:
        ad.eval()
    if tc_ad is not None:
        tc_ad.eval()
    if qad is not None:
        qad.eval()
    if fusion_ln is not None:
        fusion_ln.eval()

    field_map = None
    if args.role_fields != "none":
        import re
        LOC = ["stay", "step", "walk", "run"]
        DIR = ["none", "forward", "forward-right", "right", "back-right",
               "back", "back-left", "left", "forward-left"]
        TURN = ["none", "left", "right", "around"]
        POST = ["upright", "crouch", "jump"]
        DUR = ["1.5", "2.5", "3.5", "4.4"]

        def _idx(pat, vocab, t, default=0):
            m = re.search(pat, t)
            return vocab.index(m.group(1)) if m and m.group(1) in vocab else default

        field_map = {}
        if args.role_fields == "gt":
            rows = json.loads((DATA / "data_manifest.json").read_text())["rows"]
            for k, r in rows.items():
                p = r["plan"]
                field_map[k] = (LOC.index(p["locomotion"]), DIR.index(p["direction"]),
                                TURN.index(p["turn"]), POST.index(p["posture"]),
                                DUR.index(p["duration"]) if p["duration"] in DUR else 1)
        else:
            for nm in ("answer_self_train.raw.npz", "answer_self_val.raw.npz"):
                z = np.load(CACHE / "textfeat" / nm, allow_pickle=True)
                for k, t in zip(z["ids"], z["texts"]):
                    t = str(t)
                    field_map[str(k)] = (
                        _idx(r"locomotion=([a-z\-]+)", LOC, t, 0),
                        _idx(r"direction=([a-z\-]+)", DIR, t, 0),
                        _idx(r"turn=([a-z\-]+)", TURN, t, 0),
                        _idx(r"posture=([a-z\-]+)", POST, t, 0),
                        _idx(r"duration=([0-9.]+)", DUR, t, 1))
        print(json.dumps({"role_fields": args.role_fields, "n_fields": len(field_map)}), flush=True)

    def plan_ids(ks):
        return torch.tensor([field_map.get(k, (0, 0, 0, 0, 1)) for k in ks],
                            dtype=torch.long, device=dev)

    tc_g = tc_m = None
    tc_g_res = tc_m_res = None
    if args.cond_source in ("tcclip", "fusion"):
        z = np.load(args.tc_val)
        tc_g = {str(k): g.astype(np.float32) for k, g in zip(z["ids"], z["g"])}
        tc_m = {str(k): m.astype(np.float32) for k, m in zip(z["ids"], z["m"])}
        if args.tc_val_res:
            if args.cond_source != "tcclip":
                raise SystemExit("--tc-val-res requires --cond-source tcclip")
            zr = np.load(args.tc_val_res)
            if set(map(str, zr["ids"])) != set(tc_g):
                raise SystemExit("--tc-val-res id set differs from --tc-val")
            tc_g_res = {str(k): g.astype(np.float32) for k, g in zip(zr["ids"], zr["g"])}
            tc_m_res = {str(k): m.astype(np.float32) for k, m in zip(zr["ids"], zr["m"])}
            for name, arr in (("g", tc_g_res), ("m", tc_m_res)):
                if not all(np.isfinite(v).all() for v in arr.values()):
                    raise SystemExit(f"--tc-val-res {name} is not finite")
            print(json.dumps({"tc_val_res": args.tc_val_res, "n": len(tc_g_res)}), flush=True)

    def cond_batch(ks, stage="base"):
        if args.cond_source in ("tcclip", "fusion"):
            if stage == "res" and tc_g_res is not None:
                g = torch.from_numpy(np.stack([tc_g_res[k] for k in ks])).to(dev)
                m = torch.from_numpy(np.stack([tc_m_res[k] for k in ks])).to(dev)
                return g, m
            g = torch.from_numpy(np.stack([tc_g[k] for k in ks])).to(dev)
            m = torch.from_numpy(np.stack([tc_m[k] for k in ks])).to(dev)
            if tc_ad is not None:
                g, m = tc_ad(g, m)
            if qad is not None:
                Hf, mk = HT.visual_batch(vmeta, ks, dev)
                gq, mq = qad(Hf, mk)
                if fusion_gate is not None:
                    a = torch.tanh(fusion_gate)
                    return g + a * gq, m
                return g + gq, torch.cat([m, mq], 1)
            return g, m
        Hf, mk = batch(ks)
        if args.adapter_type == "ordered":
            Hv, mv = HT.visual_batch(vmeta_vis, ks, dev)
            return ad(Hf, mk, Hv, mv)
        if args.adapter_type in ("role", "perceiver4", "flat6m"):
            Hv, mv = HT.visual_batch(vmeta_vis, ks, dev)
            if args.adapter_type == "role" and field_map is not None:
                g, m = ad(Hf, mk, Hv, mv, plan_ids(ks))
            else:
                g, m = ad(Hf, mk, Hv, mv)
            if proj_m is not None:
                m = torch.cat([m, proj_m(m.mean(1))[:, None]], 1)
            return g, m
        g, m = ad(Hf, mk)
        if proj_m is not None:
            m = torch.cat([m, proj_m(m.mean(1))[:, None]], 1)
        return g, m

    corpus = args.corpus or ("a_gt3p_trainval" if args.cond == "video" else "gt3p_trainval")
    if args.repr == "frame":
        _fr, _mk, _ids = [], [], []
        for _p in sorted(Path(args.repr_cache).glob(f"{args.repr_name}.shard*.npz")):
            _d = np.load(_p, allow_pickle=True)
            _fr.append(_d["frame"]); _mk.append(_d["frame_mask"]); _ids += [str(x) for x in _d["ids"]]
        _fr = np.concatenate(_fr, 0); _mk = np.concatenate(_mk, 0); _row = {k: i for i, k in enumerate(_ids)}
        print(json.dumps({"repr": "frame", "n": len(_ids)}), flush=True)
    else:
        meta = json.loads((CACHE / "textfeat" / f"{corpus}.meta.json").read_text())
        feats = np.load(CACHE / "textfeat" / f"{corpus}.feats.npy", mmap_mode="r")
        tid = {x["id"]: x["tid"] for x in meta["items"]}
    if args.retr_file:
        _rd = np.load(args.retr_file, allow_pickle=True)
        _rmap = {str(k): i for i, k in enumerate(_rd["ids"])}; _remb = _rd["emb"]
        def retr_batch(keys):
            return torch.from_numpy(np.stack([_remb[_rmap[k]] for k in keys])).float().to(dev)
    split = json.loads((DATA / "split.json").read_text())
    keys = [str(x) for x in (split["train_real"] if args.split == "train" else split[args.split])]
    if args.keys_file:
        raw = Path(args.keys_file).read_text()
        try:
            want = {str(x) for x in json.loads(raw)}
        except json.JSONDecodeError:
            want = {line.strip() for line in raw.splitlines() if line.strip()}
        keys = [k for k in keys if k in want]
        print(json.dumps({"keys_file": args.keys_file, "n": len(keys)}), flush=True)
    if args.limit:
        keys = keys[:args.limit]
    zz = np.load(DATA / "motion/humanml263_full.npz")
    lpos = {str(x): i for i, x in enumerate(zz["ids"])}; zlen = zz["lengths"]
    L = np.array([min(int(zlen[lpos[k]]), 196) // 4 * 4 for k in keys], np.int64)
    if args.lengths_file:
        lf = np.load(args.lengths_file)
        lmap = {str(k): int(v) for k, v in zip(lf["ids"], lf["lengths"])}
        L = np.array([max(4, min(int(lmap.get(k, 64)), 196) // 4 * 4) for k in keys], np.int64)
        print(json.dumps({"lengths_file": args.lengths_file, "mean_len": float(L.mean())}), flush=True)

    def batch(ks):
        if args.repr == "frame":
            H = np.zeros((len(ks), _fr.shape[1], _fr.shape[2], _fr.shape[3]), np.float16)
            m = np.zeros((len(ks), _fr.shape[1]), bool)
            for i, k in enumerate(ks):
                r = _row[k]; H[i] = _fr[r]; m[i] = _mk[r]
            return torch.from_numpy(H).to(dev), torch.from_numpy(m).to(dev)
        tids = [tid[k] for k in ks]; Tm = max(meta["lengths"][t] for t in tids)
        H = np.zeros((len(tids), Tm, 4, 2560), np.float16); m = np.zeros((len(tids), Tm), bool)
        for i, t in enumerate(tids):
            n = meta["lengths"][t]; H[i, :n] = feats[meta["offsets"][t]:meta["offsets"][t] + n]; m[i, :n] = True
        return torch.from_numpy(H).to(dev), torch.from_numpy(m).to(dev)

    J = np.zeros((args.samples, len(keys), 208, 22, 3), np.float32)
    C = np.full((args.samples, len(keys), 52, Q), -1, np.int16)
    B = np.full((args.samples, len(keys), 52), -1, np.int16) if args.save_base_codes else None
    t_base = t_res = t_vq = 0.0
    n_calls = 0
    t_gen0 = time.perf_counter()
    with torch.no_grad():
        for s in range(args.samples):
            for j in range(0, len(keys), 48):
                ks = keys[j:j + 48]; g, m = cond_batch(ks)
                g_res, m_res = cond_batch(ks, "res") if tc_g_res is not None else (g, m)
                if args.ablate_g == "zero":
                    g = torch.zeros_like(g)
                elif args.ablate_g == "shuffle":
                    g = g[torch.randperm(g.shape[0], device=g.device)]
                elif args.ablate_g == "mean":
                    g = g.mean(0, keepdim=True).expand_as(g)
                if args.ablate_m == "zero":
                    m = torch.zeros_like(m)
                elif args.ablate_m == "shuffle":
                    m = m[torch.randperm(m.shape[0], device=m.device)]
                if args.retr_file:
                    m = torch.cat([m, proj(retr_batch(ks))[:, None]], 1)
                Li = [max(1, int(L[j + i]) // 4) for i in range(len(ks))]
                torch.manual_seed(1000 + s)
                _t = time.perf_counter()
                base = trans.generate(g, torch.tensor(Li, device=dev), timesteps=args.timesteps,
                                      cond_scale=args.cond_scale, temperature=args.temperature, topk_filter_thres=args.topk,
                                      gsample=False, force_mask=False, memory=m)
                t_base += time.perf_counter() - _t; n_calls += 1
                if B is not None:
                    for i in range(len(ks)):
                        n = Li[i]
                        B[s, j + i, :n] = base[i, :n].long().cpu().numpy().astype(np.int16)
                _t = time.perf_counter()
                codes = res.generate(base, g_res, torch.tensor(Li, device=dev), temperature=args.temperature,
                                     cond_scale=args.res_scale, topk_filter_thres=args.topk, memory=m_res)
                t_res += time.perf_counter() - _t
                _t = time.perf_counter()
                for i in range(len(ks)):
                    n = Li[i]
                    toks = codes[i, :n].long().clamp(0, nb - 1)  # (n, Q)
                    if args.save_codes:
                        C[s, j + i, :n] = toks.cpu().numpy().astype(np.int16)
                    rec = vq.forward_decoder(toks[None]).cpu().numpy()[0]
                    joints = HT.FH.to_canonical_joints(rec, mean, std)[:n * 4]
                    J[s, j + i, :min(len(joints), 208)] = joints[:208]
                t_vq += time.perf_counter() - _t
            print(json.dumps({"sample": s}), flush=True)
    if args.time_json:
        payload = {"n_clips": len(keys), "samples": args.samples, "batch": 48,
                   "n_calls": n_calls,
                   "gen_total_s": time.perf_counter() - t_gen0,
                   "base_s_per_call": t_base / max(n_calls, 1),
                   "res_s_per_call": t_res / max(n_calls, 1),
                   "vq_s_per_clip": t_vq / max(args.samples * len(keys), 1)}
        Path(args.time_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.time_json).write_text(json.dumps(payload, indent=1))
        print(json.dumps({"timing": payload}), flush=True)
    valid = np.arange(208)[None, :] < L[:, None]
    tmp = Path("/tmp") / f"{Path(args.out).name}.{os.getpid()}.npz"
    if args.save_codes:
        extra = {"base_codes": B} if B is not None else {}
        np.savez(tmp, ids=np.array(keys), codes=C, lengths=L, fps=np.array(20),
                 val_joints=J, valid=valid, **extra)
    else:
        extra = {"base_codes": B} if B is not None else {}
        np.savez(tmp, val_ids=np.array(keys), val_joints=J, lengths=L, valid=valid,
                 fps=np.array(20), **extra)
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    import shutil; shutil.copyfile(tmp, outp); tmp.unlink()
    print(json.dumps({"saved": str(outp), "shape": list(J.shape)}), flush=True)


if __name__ == "__main__":
    main()

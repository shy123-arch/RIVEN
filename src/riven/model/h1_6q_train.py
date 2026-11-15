#!/usr/bin/env python3
"""ViMo-free H1 with a 6-quantizer RVQ tokenizer.

Base masked transformer predicts quantizer 1 (q1); a ResidualTransformer
predicts q2..q6 conditioned on the accumulated codebook embeddings, the video
(text) condition g and the memory m.  Both are trained jointly, exactly the
MoMask/HERO recipe but with our ViMo-free VQ and a single condition path.
"""
import argparse, json, os, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h1_train as HT
import cond_adapters as CA
from motion_windows import choose_span

torch = HT.torch; dev = HT.dev; HERE = HT.HERE; DATA = HT.DATA; CACHE = HT.CACHE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vq", default=os.environ.get("RIVEN_VQ", str(HT.CACHE / "vq/best.pt")))
    ap.add_argument("--init-h1", default="")
    ap.add_argument("--init-adapter-only", action="store_true", help="load only adapter from --init-h1")
    ap.add_argument("--init-head-only", action="store_true", help="load only trans/res from --init-h1")
    ap.add_argument("--cond", default="video", choices=["video", "text"])
    ap.add_argument("--slots", type=int, default=16)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--crop-mode", choices=["random", "prefix", "full"], default="random",
                    help="random: original window; prefix: window from frame0; full: complete motion from frame0")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20261101)
    ap.add_argument("--train-frac", type=float, default=1.0)
    ap.add_argument("--cfg-mem", action="store_true", help="drop memory too in the CFG uncond branch")
    ap.add_argument("--intent-mode", default="cross", choices=["cross", "direct", "refresh"],
                    help="cross: HERO g'=CrossAttn(g,m); direct: g never reads m; "
                         "refresh: g is rewritten from the current motion state each denoising call")
    ap.add_argument("--mem-slot-drop", type=float, default=0.0,
                    help="training-time per-slot dropout of the evidence memory")
    ap.add_argument("--drop-mode", default="joint", choices=["joint", "independent", "g_only", "learned_null"],
                    help="joint: cfg_mem drops g and m together; independent: separate masks; "
                         "g_only: HERO original (drop g, never m); learned_null: replace r/e with "
                         "learnable nulls before _refine (four branches in training)")
    ap.add_argument("--null-branch", default="full",
                    choices=["full", "r0", "e0", "both"],
                    help="train-time learned-null branch (requires --drop-mode learned_null); "
                         "e0 keeps RIV only, r0 keeps AEM only, both removes both")
    ap.add_argument("--riv-prefix", default="cross", choices=["cross", "residual", "direct"],
                    help="cross: c=CrossAttn(r,e) (original); residual: c=r+CrossAttn(r,e); "
                         "direct: c=r (AEM is KV memory only)")
    ap.add_argument("--cond-drop", type=float, default=0.1,
                    help="per-branch condition dropout probability")
    ap.add_argument("--repr", default="textfeat", choices=["textfeat", "frame", "visual"])
    ap.add_argument("--repr-name", default="sft")
    ap.add_argument("--corpus", default="", help="override textfeat corpus name (e.g. blank_gt3p_trainval)")
    ap.add_argument("--repr-cache", default=os.environ.get("RIVEN_TEXTFEAT", str(HT.CACHE / "textfeat")))
    ap.add_argument("--visual-cache", default=os.environ.get("RIVEN_VISUALFEAT", str(HT.CACHE / "visualfeat")))
    ap.add_argument("--visual-name", default="visual_trainval")
    ap.add_argument("--use-layers", default="36")
    ap.add_argument("--adapter-type", default="perceiver",
                    choices=["perceiver", "raw", "two_stream", "prompt_ordered", "perceiver2",
                             "perceiver4", "flat6m", "idq", "role"])
    ap.add_argument("--role-swap", action="store_true",
                    help="role-swap control: prefix from visual tokens, memory from prompt tokens")
    ap.add_argument("--role-inverse", action="store_true",
                    help="inverse role swap: pooled AEM prefix, expanded RIV memory")
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
    ap.add_argument("--riv-pool", default="mean", choices=["mean", "attn", "last"])
    ap.add_argument("--riv-probe", action="store_true",
                    help="add a training-only plan-field CE probe on RIV (no inference input)")
    ap.add_argument("--riv-probe-weight", type=float, default=0.0)
    ap.add_argument("--n-enc-layers", type=int, default=1)
    ap.add_argument("--retr-file", default="")
    ap.add_argument("--tc-train", default="", help="precomputed TC-CLIP g/m npz for TRAIN")
    ap.add_argument("--tc-val", default="", help="precomputed TC-CLIP g/m npz for VAL")
    ap.add_argument("--q1-weight", type=float, default=1.0)
    ap.add_argument("--res-weight", type=float, default=1.0)
    ap.add_argument("--lam-nce", type=float, default=0.0)
    ap.add_argument("--nce-on-m", action="store_true", help="align mean(m) and keep the projection as an extra memory token")
    ap.add_argument("--lam-nce-m", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--status-json", default="", help="optional aggregate-only progress JSON on local storage")
    args = ap.parse_args()
    if args.window < 4:
        ap.error("--window must be at least4, including the reference window for full-mode RNG parity")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    def status(phase, **metrics):
        if args.status_json:
            target = Path(args.status_json)
            payload = dict(phase=phase, crop_mode=args.crop_mode, steps=args.steps,
                           test_enabled=False, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **metrics)
            temp = target.with_name(target.name + '.tmp')
            temp.write_text(json.dumps(payload, indent=2)); temp.replace(target)
    status('initializing')
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)

    ck = torch.load(args.vq, map_location="cpu")
    mean, std = ck["mean"], ck["std"]; nb = ck["nb_code"]; Q = ck["num_quantizers"]
    print(json.dumps({"vq": str(args.vq), "nb_code": int(nb), "quantizers": int(Q)}), flush=True)
    a = SimpleNamespace(num_quantizers=Q, shared_codebook=False, quantize_dropout_prob=0.0,
                        quantize_dropout_cutoff_index=0, nb_code=nb, code_dim=512, mu=0.99)
    from models.vq.model import RVQVAE
    vq = RVQVAE(a, 263, nb, 512, 512, 2, 2, 512, 3, 3, "relu", None).to(dev)
    vq.load_state_dict(ck["vq_model"]); vq.eval()
    for p in vq.parameters():
        p.requires_grad_(False)

    z = np.load(DATA / "motion/humanml263_full.npz")
    ids = [str(x) for x in z["ids"]]; feats = z["features"]; lens = z["lengths"]
    pos = {k: i for i, k in enumerate(ids)}
    split = json.loads((DATA / "split.json").read_text())
    tr_v = [str(x) for x in split["train_real"]]; va_v = [str(x) for x in split["val"]]
    if args.train_frac < 1.0:
        perm = np.random.default_rng(20261027).permutation(len(tr_v))
        n_keep = max(64, int(round(len(tr_v) * args.train_frac)))
        tr_v = [tr_v[i] for i in sorted(perm[:n_keep])]
        print(json.dumps({"train_frac": args.train_frac, "train_videos": len(tr_v)}), flush=True)

    corpus = args.corpus or ("a_gt3p_trainval" if args.cond == "video" else "gt3p_trainval")
    vmeta = None
    if args.repr == "visual":
        vmeta = HT.load_visual(args.visual_name, args.visual_cache)
        print(json.dumps({"repr": "visual", "n": len(vmeta["ids"]), "shape": list(vmeta["feats"].shape)}), flush=True)
        def _feat_batch(keys):
            return HT.visual_batch(vmeta, keys, dev)
    elif args.repr == "frame":
        import numpy as _np
        _fr, _mk, _ids = [], [], []
        for _p in sorted(Path(args.repr_cache).glob(f"{args.repr_name}.shard*.npz")):
            _d = _np.load(_p, allow_pickle=True)
            _fr.append(_d["frame"]); _mk.append(_d["frame_mask"]); _ids += [str(x) for x in _d["ids"]]
        _fr = _np.concatenate(_fr, 0); _mk = _np.concatenate(_mk, 0); _row = {k: i for i, k in enumerate(_ids)}
        print(json.dumps({"repr": "frame", "n": len(_ids)}), flush=True)
        def _feat_batch(keys):
            H = _np.zeros((len(keys), _fr.shape[1], _fr.shape[2], _fr.shape[3]), _np.float16)
            m = _np.zeros((len(keys), _fr.shape[1]), bool)
            for i, k in enumerate(keys):
                r = _row[k]; H[i] = _fr[r]; m[i] = _mk[r]
            return torch.from_numpy(H).to(dev), torch.from_numpy(m).to(dev)
    else:
        vmeta, vf = HT.load_tf(corpus); vpos = {x["id"]: x["tid"] for x in vmeta["items"]}
        def _feat_batch(keys):
            return HT.tf_batch(vmeta, vf, [vpos[k] for k in keys], dev)
    print(json.dumps({"cond": args.cond, "corpus": corpus, "repr": args.repr}), flush=True)
    if args.retr_file:
        _rd = np.load(args.retr_file, allow_pickle=True)
        _rmap = {str(k): i for i, k in enumerate(_rd["ids"])}
        _remb = _rd["emb"]
        def retr_batch(keys):
            return torch.from_numpy(np.stack([_remb[_rmap[k]] for k in keys])).float().to(dev)

    codes = {}
    with torch.no_grad():
        for k in tr_v + va_v:
            i = pos[k]; L = int(lens[i]) // 4 * 4
            x = torch.from_numpy(((feats[i, :L] - mean) / std).astype(np.float32))[None].to(dev)
            c, _ = vq.encode(x)
            codes[k] = c[0].cpu().numpy()  # (L, Q)
    print(json.dumps({"codes": len(codes)}), flush=True)

    class Readout(nn.Module):
        def __init__(self, d=512, out=256):
            super().__init__()
            self.net = nn.Sequential(nn.LayerNorm(d * 2), nn.Linear(d * 2, d), nn.GELU(), nn.Linear(d, out))
        def forward(self, g, m):
            return self.net(torch.cat([g, m.mean(1)], -1))

    readout = Readout(512, 256).to(dev)
    zc = np.load(Path(os.environ.get("RIVEN_TMR_MOTION_EMB", HT.CACHE / "tmr_motion_emb.npz")))
    zpos = {str(k): i for i, k in enumerate(zc["ids"])}; Zall = torch.tensor(zc["emb"])
    def nce(zh, zb, tau=0.07):
        a = torch.nn.functional.normalize(zh, dim=-1); b = torch.nn.functional.normalize(zb, dim=-1)
        return torch.nn.functional.cross_entropy(a @ b.T / tau, torch.arange(len(b), device=b.device))

    opt = SimpleNamespace(num_tokens=nb, num_quantizers=Q, device=dev)
    import models.mask_transformer.transformer_memo_cross as TMC
    if args.cfg_mem:
        import cfg_patch; cfg_patch.apply(TMC)
        print(json.dumps({"cfg_mem": True}), flush=True)
    import intent_patch; intent_patch.apply(TMC)
    if args.drop_mode == "independent":
        intent_patch.patch_independent_dropout(TMC)
    elif args.drop_mode == "g_only":
        intent_patch.patch_g_only_dropout(TMC)
    elif args.drop_mode == "learned_null":
        intent_patch.patch_learned_null(TMC)
    print(json.dumps({"intent_mode": args.intent_mode, "mem_slot_drop": args.mem_slot_drop,
                      "drop_mode": args.drop_mode}), flush=True)
    trans = TMC.MaskTransformer(code_dim=512, cond_mode="video", latent_dim=384, ff_size=1024,
                                num_layers=6, num_heads=6, dropout=0.2, clip_dim=512,
                                cond_drop_prob=args.cond_drop, opt=opt).to(dev)
    res = TMC.ResidualTransformer(code_dim=512, cond_mode="video", latent_dim=384, ff_size=1024,
                                  num_layers=6, num_heads=6, dropout=0.2, clip_dim=512,
                                  cond_drop_prob=args.cond_drop,
                                  shared_codebook=False, opt=opt).to(dev)
    layers = tuple(int(x) for x in args.use_layers.split(","))
    if args.adapter_type == "raw":
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
        ad = CA.RoleAdapter(g_layer=args.g_layer, v_layer=args.v_layer, p_layer=args.p_layer,
                            n_slots=args.slots, n_plan=args.plan_slots,
                            use_visual=args.role_mode in ("both", "visual"),
                            use_plan=args.role_mode in ("both", "plan"),
                            use_type=not args.role_no_type, use_time=not args.role_no_time,
                            n_visual=args.n_visual, use_fields=args.role_fields != "none",
                            riv_pool=args.riv_pool, swap=args.role_swap, inverse=args.role_inverse).to(dev)
    elif args.repr == "visual":
        ad = HT.VisualAdapter(use_layers=layers, n_slots=args.slots).to(dev)
    else:
        ad = HT.Adapter(use_layers=layers, n_slots=args.slots).to(dev)
    proj = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 512)).to(dev)
    proj_m = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 512)).to(dev)
    vismeta_role = HT.load_visual(args.visual_name, args.visual_cache) \
        if args.adapter_type in ("role", "perceiver4", "flat6m") else None

    field_map = None
    if args.role_fields != "none" or args.riv_probe:
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
        if args.role_fields == "gt" or (args.riv_probe and args.role_fields == "none"):
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

    def plan_batch(keys):
        return torch.tensor([field_map.get(k, (0, 0, 0, 0, 1)) for k in keys],
                            dtype=torch.long, device=dev)

    tc_g = tc_m = None
    tc_row = {}
    if args.tc_train:
        _ids, _gs, _ms = [], [], []
        for _p in (args.tc_train, args.tc_val):
            _z = np.load(_p)
            _ids += [str(x) for x in _z["ids"]]
            _gs.append(_z["g"]); _ms.append(_z["m"])
        tc_g = torch.from_numpy(np.concatenate(_gs, 0)).float().to(dev)
        tc_m = torch.from_numpy(np.concatenate(_ms, 0)).float().to(dev)
        tc_row = {k: i for i, k in enumerate(_ids)}
        print(json.dumps({"tc_train": args.tc_train, "tc_val": args.tc_val, "tc_items": len(_ids)}), flush=True)

    def apply_ad(keys):
        if tc_g is not None:
            idx = torch.tensor([tc_row[k] for k in keys], device=dev)
            return tc_g[idx], tc_m[idx]
        Hf, mk = _feat_batch(keys)
        if args.adapter_type in ("role", "perceiver4", "flat6m"):
            Hv, mv = HT.visual_batch(vismeta_role, keys, dev)
            if args.adapter_type == "role" and args.role_fields != "none" and field_map is not None:
                return ad(Hf, mk, Hv, mv, plan_batch(keys))
            return ad(Hf, mk, Hv, mv)
        return ad(Hf, mk)
    if args.init_h1:
        st0 = torch.load(args.init_h1, map_location="cpu")
        if not args.init_adapter_only:
            trans.load_state_dict(st0["trans"]); res.load_state_dict(st0["res"])
        if not args.init_head_only:
            ad.load_state_dict(st0["adapter"])
        if args.retr_file and "proj" in st0:
            proj.load_state_dict(st0["proj"])
        print(json.dumps({"init_h1": args.init_h1, "from_step": st0.get("step"),
                          "adapter_only": args.init_adapter_only, "head_only": args.init_head_only}), flush=True)
    intent_patch.configure(trans, args.intent_mode, args.mem_slot_drop)
    intent_patch.configure(res, args.intent_mode, args.mem_slot_drop)
    trans.riv_prefix = res.riv_prefix = args.riv_prefix
    if args.drop_mode == "learned_null":
        intent_patch.configure_learned_null(trans)
        intent_patch.configure_learned_null(res)
        intent_patch.set_null_branch(trans, args.null_branch)
        intent_patch.set_null_branch(res, args.null_branch)
        print(json.dumps({"null_branch": args.null_branch}), flush=True)
    riv_heads = None
    if args.riv_probe:
        riv_heads = nn.ModuleList([nn.Linear(512, n) for n in (4, 9, 4, 3, 4)]).to(dev)
    params = list(trans.parameters()) + list(res.parameters()) + list(ad.parameters()) + list(proj.parameters()) + list(readout.parameters()) + list(proj_m.parameters())
    if riv_heads is not None:
        params += list(riv_heads.parameters())
    optm = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.99), weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.LambdaLR(optm, lambda s: min(1.0, (s + 1) / 200) * (0.1 if s >= 0.7 * args.steps else 1.0))

    supervision = dict(valid_tokens=0, padded_tokens=0, max_latent_length=0)
    def sample(bs):
        ii = rng.choice(len(tr_v), bs, replace=False)
        arrs, Lc, keys = [], [], []
        for j in ii:
            k = tr_v[j]; n = int(lens[pos[k]]) // 4 * 4
            s, w = choose_span(n, args.window, args.crop_mode, rng)
            arrs.append(codes[k][s // 4:(s + w) // 4]); Lc.append(w // 4); keys.append(k)
        supervision['valid_tokens'] += sum(Lc)
        supervision['padded_tokens'] += bs * max(Lc)
        supervision['max_latent_length'] = max(supervision['max_latent_length'], max(Lc))
        c = torch.zeros((bs, max(Lc), Q), dtype=torch.long)
        for b, arr in enumerate(arrs):
            c[b, :len(arr)] = torch.from_numpy(arr.astype(np.int64))
        g, m = apply_ad(keys)
        if args.retr_file:
            m = torch.cat([m, proj(retr_batch(keys))[:, None]], 1)
        return c.to(dev), g, torch.tensor(Lc, device=dev), m, keys

    @torch.no_grad()
    def val_ce(n=128):
        trans.eval(); res.eval(); ad.eval()
        tb = rb = 0.0; cnt = 0
        for j in range(0, min(n, len(va_v)), 64):
            ks = va_v[j:j + 64]
            Ls = [len(codes[k]) for k in ks]
            c = torch.zeros((len(ks), max(Ls), Q), dtype=torch.long)
            for b, k in enumerate(ks):
                c[b, :len(codes[k])] = torch.from_numpy(codes[k].astype(np.int64))
            g, m = apply_ad(ks)
            if args.retr_file:
                m = torch.cat([m, proj(retr_batch(ks))[:, None]], 1)
            l1, _, _ = trans(c[..., 0].to(dev), g, torch.tensor(Ls, device=dev), m)
            l2, _, _ = res(c.to(dev), g, torch.tensor(Ls, device=dev), m)
            tb += float(l1) * len(ks); rb += float(l2) * len(ks); cnt += len(ks)
        trans.train(); res.train(); ad.train()
        return tb / cnt, rb / cnt

    log = open(out / "train.jsonl", "w"); t0 = time.time()
    for step in range(args.steps + 1):
        if step > 0:
            c, g, Ls, m, keys = sample(args.batch)
            l1, _, acc1 = trans(c[..., 0], g, Ls, m)
            l2, _, acc2 = res(c, g, Ls, m)
            loss = args.q1_weight * l1 + args.res_weight * l2
            probe = torch.zeros((), device=dev)
            if args.riv_probe and args.riv_probe_weight > 0 and riv_heads is not None:
                pid = plan_batch(keys)
                probe = sum(nn.functional.cross_entropy(riv_heads[i](g), pid[:, i]) for i in range(5))
                loss = loss + args.riv_probe_weight * probe
            aux = torch.zeros((), device=dev)
            if args.lam_nce > 0:
                zh = readout(g, m); zt = Zall[[zpos[k] for k in keys]].to(dev)
                aux = nce(zh, zt)
                zc_ = zh - zh.mean(0); var = torch.relu(1.0 - torch.sqrt(zc_.var(0) + 1e-4)).mean()
                cov = zc_.T @ zc_ / max(1, len(zc_) - 1); off = cov - torch.diag(torch.diag(cov))
                aux = aux + 0.1 * var + 0.05 * (off ** 2).sum() / 256
                loss = loss + args.lam_nce * aux
            if args.nce_on_m and args.lam_nce_m > 0:
                zh512 = proj_m(m.mean(1)); zh_m = zh512[:, :256]
                zt_m = Zall[[zpos[k] for k in keys]].to(dev)
                aux_m = nce(zh_m, zt_m)
                zc_m = zh_m - zh_m.mean(0)
                var_m = torch.relu(1.0 - torch.sqrt(zc_m.var(0) + 1e-4)).mean()
                cov_m = zc_m.T @ zc_m / max(1, len(zc_m) - 1)
                off_m = cov_m - torch.diag(torch.diag(cov_m))
                aux_m = aux_m + 0.1 * var_m + 0.05 * (off_m ** 2).sum() / 256
                loss = loss + args.lam_nce_m * aux_m
                aux = aux + aux_m
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite training loss')
            optm.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); optm.step(); sched.step()
        if step % args.eval_every == 0:
            v1, v2 = val_ce()
            r = {"step": step, "val_ce_q1": round(v1, 5), "val_ce_res": round(v2, 5), "s": round(time.time() - t0)}
            if step > 0:
                r.update({"tr_ce_q1": float(l1), "tr_ce_res": float(l2), "tr_acc_q1": float(acc1), "tr_acc_res": float(acc2), "tr_aux": float(aux), "tr_probe": float(probe)})
            log.write(json.dumps(r) + "\n"); log.flush(); print(json.dumps(r), flush=True)
            status('training', **r, **supervision)
        if step > 0 and step % args.save_every == 0:
            torch.save({"trans": HT.FH.no_clip(trans.state_dict()), "res": HT.FH.no_clip(res.state_dict()),
                        "adapter": ad.state_dict(), "proj": proj.state_dict(), "proj_m": proj_m.state_dict(),
                        "readout": readout.state_dict(), "vq": str(args.vq), "step": step,
                        "riv_heads": riv_heads.state_dict() if riv_heads is not None else {},
                        "args": vars(args)},
                       out / f"step_{step}.pt")
    torch.save({"trans": HT.FH.no_clip(trans.state_dict()), "res": HT.FH.no_clip(res.state_dict()),
                "adapter": ad.state_dict(), "proj": proj.state_dict(), "proj_m": proj_m.state_dict(),
                "readout": readout.state_dict(), "vq": str(args.vq), "step": args.steps,
                "riv_heads": riv_heads.state_dict() if riv_heads is not None else {},
                "args": vars(args)},
               out / "last.pt")
    status('complete', step=args.steps, elapsed_seconds=time.time()-t0, **supervision)


if __name__ == "__main__":
    main()

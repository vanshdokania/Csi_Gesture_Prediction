#!/usr/bin/env python3
"""
train_percom.py - our first pose model: the instructor's SiWiSNet trained and evaluated on all
prepared recordings. This is the "earlier model (without speed)" of the report: it only uses the
two signal-strength-spread channels, the repository's own loss and Adam with the repository
learning rate, 50 epochs, batch size 16.

See train_percom_v2.py for the improved version we used afterwards.

    python train_percom.py --data ~/percom/siwis_data --split person --test_person Vansh
    python train_percom.py --data ~/percom/siwis_data --split person --loso        # every person unseen once
    python train_percom.py --data ~/percom/siwis_data --split chrono               # the paper's 80/20 protocol

Models, losses and the dataset class are the instructor's and are used unchanged. Only the data
loading is ours.

Splits
    person  all recordings of --test_person are held out and never seen in training. This is the real
            test of whether the model works for a new person.
    chrono  the paper's protocol (section 4.2) applied to every recording: the first 80 % of its time is
            used for training and the last 20 % for testing. The same person, room and session appear in
            both parts, so this is much easier. Report it as the "paper protocol", not as generalisation.

Besides mask mAP and PCK (as in the repository's evaluate.py) it prints three extra things:
  - a mean-pose baseline (predict the average training mask and skeleton), because the repository README
    itself notes that a centred average skeleton already scores quite well,
  - the PCK on the two wrists only (the joints that matter for a raised hand),
  - a breakdown per gesture (bhu / lhu / rhu).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

import config
import percom_common as pc
from losses.losses import total_loss
from models.siwis_net import SiWiSNet


def load_all(data, exclude):
    recs = []
    for m in sorted((Path(data) / "rec").glob("*/meta.json")):
        import json
        meta = json.loads(m.read_text())
        if meta["person"].lower() in {e.lower() for e in exclude}:
            continue
        meta["ds"] = pc.load_recording(m.parent)
        recs.append(meta)
    if not recs:
        raise SystemExit(f"no prepared recordings in {data}/rec - run prepare_percom.py first")
    return recs


def make_split(recs, split, test_person):
    """-> train pieces, test pieces: lists of (meta, dataset, indices)."""
    tr, te = [], []
    for r in recs:
        n = len(r["ds"])
        if split == "person":
            (te if r["person"] == test_person else tr).append((r, r["ds"], list(range(n))))
        else:
            cut = int(n * config.TRAIN_FRACTION)
            tr.append((r, r["ds"], list(range(cut))))
            te.append((r, r["ds"], list(range(cut, n))))
    return tr, te


def to_loader(pieces, bs, shuffle):
    ds = ConcatDataset([Subset(d, idx) for _, d, idx in pieces])
    return DataLoader(ds, batch_size=bs, shuffle=shuffle)


@torch.no_grad()
def predict(model, pieces, bs, dev):
    """Per-piece predictions -> dict of concatenated arrays + activity / person tags per frame."""
    out = dict(pm=[], gm=[], pp=[], gc=[], gv=[], act=[], person=[])
    model.eval()
    for meta, ds, idx in pieces:
        for i in range(0, len(idx), bs):
            items = [ds[j] for j in idx[i:i + bs]]
            sig = torch.stack([it["signal"] for it in items]).to(dev)
            o = model(sig)
            out["pm"].append(torch.sigmoid(o["mask"])[:, 0].cpu().numpy())
            out["pp"].append(o["pose_heatmaps"].cpu().numpy())
            out["gm"].append(np.stack([it["mask_label"][0].numpy() for it in items]))
            out["gc"].append(np.stack([it["keypoint_coords"][0].numpy() for it in items]))
            out["gv"].append(np.stack([it["keypoint_visible"][0].numpy() for it in items]))
            out["act"] += [meta["activity"]] * len(items)
            out["person"] += [meta["person"]] * len(items)
    for k in ("pm", "gm", "pp", "gc", "gv"):
        out[k] = np.concatenate(out[k])
    out["act"], out["person"] = np.array(out["act"]), np.array(out["person"])
    return out


def score(pm, gm, pred_pts, gc, gv):
    m_ap, ap = pc.mask_map(pm, gm)
    p_all, n_all = pc.pck(pred_pts, gc.astype(np.float64), gv)
    p_wr, n_wr = pc.pck(pred_pts, gc.astype(np.float64), gv, joints=(pc.LEFT_WRIST, pc.RIGHT_WRIST))
    return dict(mask_mAP=m_ap, mask_AP50=ap[0.5], mask_AP70=ap[0.7], PCK10=p_all[0.1], PCK20=p_all[0.2], PCK30=p_all[0.3],
                PCK50=p_all[0.5], wrist_PCK20=p_wr[0.2], wrist_PCK30=p_wr[0.3], pck_frames=n_all, wrist_frames=n_wr)


def baseline(train_pieces, te):
    """Predict the average training mask and the average position of every joint, for every test frame."""
    masks, pts, vis = [], [], []
    for meta, ds, idx in train_pieces:
        masks.append(np.asarray(ds.mask_labels)[idx].astype(np.float32))
        pts.append(ds.keypoint_coords[idx, 0].astype(np.float64))
        vis.append(ds.keypoint_visible[idx, 0])
    masks, pts, vis = np.concatenate(masks), np.concatenate(pts), np.concatenate(vis)
    mean_mask = masks.mean(0)
    w = vis[..., None].astype(np.float64)
    mean_pts = (pts * w).sum(0) / np.maximum(w.sum(0), 1)
    n = len(te["gm"])
    return score(np.broadcast_to(mean_mask, (n,) + mean_mask.shape), te["gm"], np.broadcast_to(mean_pts, (n,) + mean_pts.shape),
                 te["gc"], te["gv"])


def fmt(d):
    return (f"mask mAP {d['mask_mAP']:.3f} AP50 {d['mask_AP50']:.3f} AP70 {d['mask_AP70']:.3f} | "
            f"PCK@0.1 {d['PCK10']:.3f} @0.2 {d['PCK20']:.3f} @0.3 {d['PCK30']:.3f} @0.5 {d['PCK50']:.3f} | "
            f"wrists PCK@0.2 {d['wrist_PCK20']:.3f} @0.3 {d['wrist_PCK30']:.3f}  (PCK frames {d['pck_frames']}, wrist {d['wrist_frames']})")


def run_fold(recs, args, test_person, dev, tag):
    tr, te = make_split(recs, args.split, test_person)
    if not tr or not te:
        raise SystemExit("empty train or test set - check --test_person / the prepared recordings")
    n_tr, n_te = sum(len(i) for _, _, i in tr), sum(len(i) for _, _, i in te)
    who = f"test person {test_person}" if args.split == "person" else "last 20% of every recording"
    print(f"\n=== {tag}: {who} | train {n_tr} frames ({len(tr)} recordings) / test {n_te} frames ===")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = SiWiSNet().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    loader = to_loader(tr, args.batch_size, True)
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        run = {"mask_loss": 0.0, "pose_loss": 0.0}
        for b in loader:
            o = model(b["signal"].to(dev))
            loss, parts = total_loss(torch.sigmoid(o["mask"]), b["mask_label"].to(dev), o["pose_heatmaps"],
                                     b["keypoint_label"].to(dev), o["pose_tags"], b["keypoint_coords"].to(dev),
                                     b["keypoint_visible"].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            for k in run:
                run[k] += parts[k]
        if ep == 0 or (ep + 1) % args.print_every == 0 or ep == args.epochs - 1:
            nb = max(len(loader), 1)
            print(f"  epoch {ep + 1:3d}  mask_loss {run['mask_loss'] / nb:10.2f}  pose_loss {run['pose_loss'] / nb:.5f}"
                  f"   ({time.time() - t0:.0f} s)", flush=True)
    pred = predict(model, te, args.batch_size, dev)
    pts = pc.argmax_points(pred["pp"])
    res = dict(model=score(pred["pm"], pred["gm"], pts, pred["gc"], pred["gv"]), baseline=baseline(tr, pred),
               by_activity={}, test_person=test_person, split=args.split, n_train=n_tr, n_test=n_te)
    print("  MODEL     :", fmt(res["model"]))
    print("  MEAN-POSE :", fmt(res["baseline"]), " <- predict the average mask/skeleton, no CSI used")
    for a in sorted(set(pred["act"])):
        s = pred["act"] == a
        res["by_activity"][a] = score(pred["pm"][s], pred["gm"][s], pts[s], pred["gc"][s], pred["gv"][s])
        print(f"    {a:4s}    :", fmt(res["by_activity"][a]))
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / f"siwis_{args.split}_{test_person if args.split == 'person' else 'chrono'}.pt")
    np.savez_compressed(out / f"pred_{args.split}_{test_person if args.split == 'person' else 'chrono'}.npz",
                        masks=pred["pm"].astype(np.float16), heatmaps=pred["pp"].astype(np.float16),
                        gt_masks=pred["gm"], activity=pred["act"], person=pred["person"])
    return res


def main():
    ap = argparse.ArgumentParser(description="Train and evaluate SiWiSNet on our recordings")
    ap.add_argument("--data", default="~/percom/siwis_data", help="folder made by prepare_percom.py")
    ap.add_argument("--out", default="~/percom/siwis_results")
    ap.add_argument("--split", choices=["person", "chrono"], default="person")
    ap.add_argument("--test_person", default="", help="person split: held-out person (default: last alphabetically)")
    ap.add_argument("--loso", action="store_true", help="person split: repeat for every person, report the mean")
    ap.add_argument("--exclude", nargs="*", default=["Divij", "Nikita"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--rate", type=float, default=150.0, help="must equal --rate used in prepare_percom.py")
    ap.add_argument("--print_every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    args.data = str(Path(args.data).expanduser())
    pc.set_sampling_rate(args.rate)
    dev = torch.device(args.device)

    recs = load_all(args.data, args.exclude)
    persons = sorted({r["person"] for r in recs})
    print(f"device {dev} | {len(recs)} recordings | persons {persons} | excluded {args.exclude}")
    print("frames per activity:", {a: sum(len(r['ds']) for r in recs if r['activity'] == a)
                                    for a in sorted({r['activity'] for r in recs})})
    results = []
    if args.split == "chrono":
        results.append(run_fold(recs, args, "", dev, "paper protocol (chronological 80/20 per recording)"))
    else:
        if len(persons) < 2:
            raise SystemExit("person split needs at least 2 persons")
        todo = persons if args.loso else [args.test_person or persons[-1]]
        for i, p in enumerate(todo, 1):
            results.append(run_fold(recs, args, p, dev, f"fold {i}/{len(todo)}"))
        if len(results) > 1:
            print("\n=== leave-one-person-out mean ===")
            keys = results[0]["model"].keys()
            for name in ("model", "baseline"):
                mean = {k: float(np.nanmean([r[name][k] for r in results])) for k in keys}
                print(f"  {name:9s}:", fmt(mean))
    pc.save_json(Path(args.out).expanduser() / f"results_{args.split}.json", results)
    print(f"\nSaved in {Path(args.out).expanduser()}")


if __name__ == "__main__":
    main()

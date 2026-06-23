"""
Paper-ready attention comparison figure.

Takes an explicit list of test-set sample IDs (the ones already validated as
"cbm+contrastive correct, sft baseline wrong"). For each sample, runs BOTH
Qwen2-VL checkpoints with output_attentions=True, computes the heatmap as the
mean attention from generated answer tokens to image-patch tokens, AVERAGED
across the last 4 LLM layers (and over heads + gen steps).

Heatmap post-processing for clean paper figures:
  - Gaussian smoothing (sigma in patch grid space) before upsampling.
  - Percentile clipping (default 5..99) to suppress single-patch outliers and
    raise mid-range contrast.
  - Bilinear upsample to image size.

Output:
  - sample_<id>.png : 1 row x 3 cols [original (with GT bbox) | SFT baseline | CBM + Contrastive]
  - _grid.png       : 8 rows x 3 cols composite for the paper.
"""

import argparse
import gc
import json
import re
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

IMAGE_TOKEN_ID = 151655
LAST_N_LAYERS = 4
MAX_NEW_TOKENS = 256
BBOX_RE = re.compile(r"```json\s*(\[.*?\])\s*```", re.S)
FINAL_ANS_RE = re.compile(r"\+\s*Final\s*Answer[:\s]*(.+?)(?:\n|$)", re.IGNORECASE)

LABEL_BASELINE = "SFT baseline"
LABEL_TARGET = "SFT + CBM + Contrastive"


def normalize_answer(answer):
    if answer is None:
        return None
    answer = answer.lower().strip().rstrip(".,;:!?")
    if not answer:
        return None
    if "moderate" in answer:
        return "moderate-dementia"
    if "mild" in answer and "non" not in answer:
        return "mild-dementia"
    if "non-dementia" in answer or "non dementia" in answer or "normal" in answer:
        return "non-dementia"
    return answer


def parse_final_answer(text):
    if not text:
        return None
    m = FINAL_ANS_RE.search(text)
    if m:
        return normalize_answer(m.group(1).strip())
    response_lower = text.lower()
    for kw, lab in [("moderate-dementia", "moderate-dementia"),
                     ("moderate dementia", "moderate-dementia"),
                     ("mild-dementia", "mild-dementia"),
                     ("mild dementia", "mild-dementia"),
                     ("non-dementia", "non-dementia"),
                     ("non dementia", "non-dementia")]:
        if kw in response_lower:
            return lab
    return None


def parse_gt_bboxes(gt_value):
    m = BBOX_RE.search(gt_value)
    if not m:
        return []
    try:
        items = json.loads(m.group(1))
    except Exception:
        return []
    out = []
    for it in items:
        b = it.get("bbox_2d")
        if isinstance(b, (list, tuple)) and len(b) == 4:
            out.append([int(v) for v in b])
    return out


def _make_sample_dict(d, image_dir):
    question = d["conversations"][0]["value"].replace("<image>", "").strip()
    gt_text = d["conversations"][1]["value"]
    return {
        "id": d["id"],
        "image": str(Path(image_dir) / d["image"]),
        "question": question,
        "gt_text": gt_text,
        "gt_label": parse_final_answer(gt_text),
        "gt_bboxes": parse_gt_bboxes(gt_text),
    }


def load_samples_by_ids(test_json, image_dir, ids):
    with open(test_json) as f:
        data = json.load(f)
    by_id = {str(d["id"]): d for d in data}
    out, missing = [], []
    for sid in ids:
        d = by_id.get(str(sid))
        if d is None:
            missing.append(sid); continue
        out.append(_make_sample_dict(d, image_dir))
    if missing:
        print(f"[warn] IDs not found in test set: {missing}")
    return out


def load_pool(test_json, image_dir, pool_size, seed):
    import random as _random
    with open(test_json) as f:
        data = json.load(f)
    rng = _random.Random(seed)
    have = [d for d in data
            if parse_gt_bboxes(d["conversations"][1]["value"])
            and parse_final_answer(d["conversations"][1]["value"]) is not None]
    picked = rng.sample(have, min(pool_size, len(have)))
    return [_make_sample_dict(d, image_dir) for d in picked]


def build_inputs_for_generation(processor, pil_image, question, device):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": question},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[text], images=[pil_image], return_tensors="pt").to(device)


def extract_avg_attn_to_image(outputs, image_positions, last_n_layers):
    """Average attention from generated tokens to image patches across the
    last N layers (and over heads + gen steps). Returns 1D float32 array."""
    n_steps = len(outputs.attentions)
    if n_steps == 0:
        return None
    n_layers_total = len(outputs.attentions[0])
    layer_idxs = list(range(n_layers_total - last_n_layers, n_layers_total))
    n_img = image_positions.numel()
    img = image_positions.cpu()

    accum = torch.zeros(n_img, dtype=torch.float32)
    counts = 0
    for t in range(n_steps):
        layer_attns = outputs.attentions[t]
        q_index = -1 if t == 0 else 0
        for li in layer_idxs:
            attn = layer_attns[li]                  # (1, H, q_len, k_len)
            a = attn[0, :, q_index, :]              # (H, k_len)
            a_img = a[:, img.to(a.device)].mean(dim=0).float().cpu()
            accum += a_img
            counts += 1
    return (accum / max(counts, 1)).numpy()


@torch.inference_mode()
def run_model(model_path, samples, device="cuda"):
    print(f"\n[attn] Loading {Path(model_path).name} (eager)")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device,
        attn_implementation="eager", trust_remote_code=True,
    ).eval()

    out = []
    for i, s in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] id={s['id']}", flush=True)
        pil = Image.open(s["image"]).convert("RGB")
        inputs = build_inputs_for_generation(processor, pil, s["question"], device)
        input_ids = inputs.input_ids[0]
        image_positions = (input_ids == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]

        thw = inputs["image_grid_thw"][0].tolist()
        merge = getattr(model.config.vision_config, "spatial_merge_size", 2)
        gh, gw = thw[1] // merge, thw[2] // merge

        gen_out = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            output_attentions=True, return_dict_in_generate=True, use_cache=True,
        )
        attn_vec = extract_avg_attn_to_image(gen_out, image_positions, LAST_N_LAYERS)
        heatmap_grid = attn_vec.reshape(gh, gw)

        gen_ids = gen_out.sequences[0, input_ids.shape[0]:]
        gen_text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
        pred_label = parse_final_answer(gen_text)

        out.append({
            "id": s["id"], "image": s["image"], "question": s["question"],
            "gt_label": s["gt_label"], "gt_bboxes": s["gt_bboxes"],
            "pred_label": pred_label, "gen_text": gen_text,
            "heatmap_grid": heatmap_grid, "image_size": pil.size,
        })

        del gen_out, inputs
        torch.cuda.empty_cache()

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return out


def postprocess_heatmap(grid, image_size_wh, sigma=1.5, percentiles=(5, 99)):
    """grid: (gh, gw); returns (H, W) in [0, 1] after smoothing + percentile clip."""
    g = grid.astype(np.float32)
    if sigma > 0:
        g = gaussian_filter(g, sigma=sigma)
    W, H = image_size_wh
    img = Image.fromarray(g).resize((W, H), resample=Image.BILINEAR)
    h = np.asarray(img, dtype=np.float32)
    if sigma > 0:
        h = gaussian_filter(h, sigma=max(2, min(W, H) / 100))
    lo, hi = np.percentile(h, percentiles)
    if hi > lo:
        h = np.clip((h - lo) / (hi - lo), 0, 1)
    else:
        h = np.zeros_like(h)
    return h


def draw_bboxes(ax, bboxes, color="lime", lw=2.5):
    for x1, y1, x2, y2 in bboxes:
        ax.add_patch(mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=lw, edgecolor=color, facecolor="none",
        ))


def render_overlay(ax, pil_image, heatmap_full, bboxes, title):
    ax.imshow(pil_image)
    ax.imshow(heatmap_full, cmap="jet", alpha=0.5)
    draw_bboxes(ax, bboxes)
    ax.set_title(title, fontsize=13)
    ax.axis("off")


CANONICAL_LABELS = {"moderate-dementia", "mild-dementia", "non-dementia"}


def display_label(pred):
    """Shorten the parsed prediction to one tidy token for paper figures.
    - canonical labels pass through (e.g. 'moderate-dementia')
    - None -> 'None'
    - garbage like 'severe</think><answer>...' -> just 'severe'
    """
    if pred is None:
        return "None"
    if pred in CANONICAL_LABELS:
        return pred
    first = re.split(r"[\s<>{}\[\]\"`,]", pred, maxsplit=1)[0]
    return first if first else "?"


def correctness_marker(pred, gt):
    if pred is None or pred not in CANONICAL_LABELS:
        return "✗"
    return "✓" if pred == gt else "✗"


def render_per_sample(base_results, targ_results, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for b, t in zip(base_results, targ_results):
        pil = Image.open(b["image"]).convert("RGB")
        b_hm = postprocess_heatmap(b["heatmap_grid"], pil.size)
        t_hm = postprocess_heatmap(t["heatmap_grid"], pil.size)

        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        axes[0].imshow(pil); draw_bboxes(axes[0], b["gt_bboxes"]); axes[0].axis("off")
        axes[0].set_title(f"id={b['id']}   GT: {b['gt_label']}   (bbox in lime)", fontsize=13)
        render_overlay(axes[1], pil, b_hm, b["gt_bboxes"],
                       f"{LABEL_BASELINE}   pred: {display_label(b['pred_label'])}   {correctness_marker(b['pred_label'], b['gt_label'])}")
        render_overlay(axes[2], pil, t_hm, t["gt_bboxes"],
                       f"{LABEL_TARGET}   pred: {display_label(t['pred_label'])}   {correctness_marker(t['pred_label'], t['gt_label'])}")
        q_short = b["question"]
        if len(q_short) > 160:
            q_short = q_short[:160] + "…"
        fig.suptitle(f"Q: {q_short}", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        out_path = out_dir / f"sample_{b['id']}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_path}")
        rows.append((b, t, b_hm, t_hm))
    return rows


def render_grid(rows, out_path):
    n = len(rows)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 3, figsize=(16, 5.2 * n))
    if n == 1:
        axes = axes[None, :]
    for r, (b, t, b_hm, t_hm) in enumerate(rows):
        pil = Image.open(b["image"]).convert("RGB")
        axes[r, 0].imshow(pil); draw_bboxes(axes[r, 0], b["gt_bboxes"]); axes[r, 0].axis("off")
        axes[r, 0].set_title(f"id={b['id']}   GT: {b['gt_label']}", fontsize=12)
        render_overlay(axes[r, 1], pil, b_hm, b["gt_bboxes"],
                       f"{LABEL_BASELINE}   pred: {display_label(b['pred_label'])}   {correctness_marker(b['pred_label'], b['gt_label'])}"
                       if r == 0 else f"pred: {display_label(b['pred_label'])}   {correctness_marker(b['pred_label'], b['gt_label'])}")
        render_overlay(axes[r, 2], pil, t_hm, t["gt_bboxes"],
                       f"{LABEL_TARGET}   pred: {display_label(t['pred_label'])}   {correctness_marker(t['pred_label'], t['gt_label'])}"
                       if r == 0 else f"pred: {display_label(t['pred_label'])}   {correctness_marker(t['pred_label'], t['gt_label'])}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_json", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--baseline_model", required=True)
    ap.add_argument("--target_model", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sample_ids", required=True,
                    help="comma-separated test-set sample IDs, e.g. 4327,2208,4407,...")
    args = ap.parse_args()

    ids = [s.strip() for s in args.sample_ids.split(",") if s.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples_by_ids(args.test_json, args.image_dir, ids)
    print(f"[main] loaded {len(samples)} samples by ID")
    for s in samples:
        print(f"  id={s['id']}  gt={s['gt_label']}  bboxes={len(s['gt_bboxes'])}")

    print("\n[main] === Running SFT baseline (cbm-only) ===")
    base = run_model(args.baseline_model, samples)
    print("\n[main] === Running SFT + CBM + Contrastive ===")
    targ = run_model(args.target_model, samples)

    print("\n[main] === Rendering ===")
    rows = render_per_sample(base, targ, out_dir)
    render_grid(rows, out_dir / "_grid.png")

    summary = []
    for b, t in zip(base, targ):
        summary.append({
            "id": b["id"], "gt_label": b["gt_label"],
            "baseline_pred": b["pred_label"], "baseline_correct": b["pred_label"] == b["gt_label"],
            "target_pred": t["pred_label"], "target_correct": t["pred_label"] == t["gt_label"],
        })
    (out_dir / "_predictions.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[main] Done. Output: {out_dir}")


if __name__ == "__main__":
    main()

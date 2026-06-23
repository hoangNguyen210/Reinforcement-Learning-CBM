"""
Compare per-layer image attention of two Qwen2-VL checkpoints (CBM-only baseline
vs CBM+Contrastive) on test samples where ONLY the CBM+Contrastive model gets
the final diagnosis right.

Pipeline:
  Pass 1 (fast, sdpa): generate answers for POOL_SIZE samples with each model,
    parse "Final Answer: <label>", score correctness vs GT.
  Filter: keep samples where cbm_contrastive == correct AND cbm_only == wrong.
    Pick first NUM_VIZ samples (default 10).
  Pass 2 (eager, output_attentions=True): for each filtered sample, generate
    with attentions enabled. For each of the LAST_N_LAYERS, take attention from
    EVERY generated answer token to the image-patch tokens (mean over heads
    and gen steps per layer; NO averaging across layers; NO concept restriction).
  Render: per sample, 4 rows x 3 cols
    [ original (with GT bbox) | cbm-only layer L | cbm+con layer L ]
    Rows are layer L = -4, -3, -2, -1 (top → last layer at the bottom).
"""

import argparse
import gc
import json
import random
import re
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

IMAGE_TOKEN_ID = 151655
LAST_N_LAYERS = 4
MAX_NEW_TOKENS = 256
BBOX_RE = re.compile(r"```json\s*(\[.*?\])\s*```", re.S)
FINAL_ANS_RE = re.compile(r"\+\s*Final\s*Answer[:\s]*(.+?)(?:\n|$)", re.IGNORECASE)


def normalize_answer(answer: str):
    """Mirror of ttrl_utils.normalize_answer (S-Chain 3-class)."""
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


def parse_final_answer(text: str):
    """Mirror of ttrl_utils.extract_final_answer for the S-Chain dementia 3-class.
    Primary regex on '+ Final Answer: <label>'; fallback to keyword-anywhere.
    """
    if not text:
        return None
    m = FINAL_ANS_RE.search(text)
    if m:
        return normalize_answer(m.group(1).strip())
    response_lower = text.lower()
    if "moderate-dementia" in response_lower or "moderate dementia" in response_lower:
        return "moderate-dementia"
    if "mild-dementia" in response_lower or "mild dementia" in response_lower:
        return "mild-dementia"
    if "non-dementia" in response_lower or "non dementia" in response_lower:
        return "non-dementia"
    return None


def parse_gt_bboxes(gt_value: str) -> list:
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


def load_pool(test_json: str, image_dir: str, pool_size: int, seed: int):
    with open(test_json) as f:
        data = json.load(f)
    rng = random.Random(seed)
    have_bbox_and_label = []
    for d in data:
        if not parse_gt_bboxes(d["conversations"][1]["value"]):
            continue
        if parse_final_answer(d["conversations"][1]["value"]) is None:
            continue
        have_bbox_and_label.append(d)
    picked = rng.sample(have_bbox_and_label, pool_size)
    out = []
    for item in picked:
        question = item["conversations"][0]["value"].replace("<image>", "").strip()
        img_path = Path(image_dir) / item["image"]
        bboxes = parse_gt_bboxes(item["conversations"][1]["value"])
        gt_text = item["conversations"][1]["value"]
        gt_label = parse_final_answer(gt_text)
        out.append({
            "id": item["id"], "image": str(img_path),
            "question": question, "gt_bboxes": bboxes,
            "gt_label": gt_label, "gt_text": gt_text,
        })
    return out


def build_inputs_for_generation(processor, pil_image: Image.Image, question: str, device):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": question},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[pil_image], return_tensors="pt").to(device)
    return inputs


@torch.inference_mode()
def predict_pass(model_path: str, samples: list, device: str = "cuda"):
    """Generate answers for every sample; return list of {id, gen_text, pred_label, correct}."""
    print(f"\n[pred] Loading {Path(model_path).name} (sdpa)")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device,
        attn_implementation="sdpa", trust_remote_code=True,
    ).eval()

    out = []
    for i, s in enumerate(samples):
        pil = Image.open(s["image"]).convert("RGB")
        inputs = build_inputs_for_generation(processor, pil, s["question"], device)
        gen = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True,
        )
        gen_ids = gen[0, inputs.input_ids.shape[1]:]
        gen_text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
        pred_label = parse_final_answer(gen_text)
        correct = (pred_label is not None and s["gt_label"] is not None
                   and pred_label.lower() == s["gt_label"].lower())
        print(f"  [{i+1}/{len(samples)}] id={s['id']:>6}  gt={s['gt_label']:<18}  pred={str(pred_label):<18}  {'✓' if correct else '✗'}", flush=True)
        out.append({"id": s["id"], "gen_text": gen_text, "pred_label": pred_label, "correct": correct})
        del inputs, gen
        torch.cuda.empty_cache()

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return out


def extract_per_layer_attn_to_image(outputs, image_positions: torch.Tensor, last_n_layers: int) -> dict:
    """outputs from generate(output_attentions=True, return_dict_in_generate=True).
    outputs.attentions: tuple of T gen steps, each is tuple of L layers.
        step 0:  (1, H, prompt_len, prompt_len)         — prefill; use last query
        step >0: (1, H, 1, prompt_len + t)              — decode; use only query
    Returns {layer_idx: 1D numpy array of len(image_positions)} for the LAST N layers,
    averaged across heads and gen steps within each layer (NO across-layer averaging).
    """
    n_steps = len(outputs.attentions)
    if n_steps == 0:
        return {}
    n_layers_total = len(outputs.attentions[0])
    layer_idxs = list(range(n_layers_total - last_n_layers, n_layers_total))
    n_img = image_positions.numel()
    img = image_positions.cpu()

    accum = {li: torch.zeros(n_img, dtype=torch.float32) for li in layer_idxs}
    counts = {li: 0 for li in layer_idxs}

    for t in range(n_steps):
        layer_attns = outputs.attentions[t]
        q_index = -1 if t == 0 else 0
        for li in layer_idxs:
            attn = layer_attns[li]                 # (1, H, q_len, k_len)
            a = attn[0, :, q_index, :]             # (H, k_len)
            a_img = a[:, img.to(a.device)].mean(dim=0).float().cpu()  # (n_img,)
            accum[li] += a_img
            counts[li] += 1
    return {li: (accum[li] / max(counts[li], 1)).numpy() for li in layer_idxs}


@torch.inference_mode()
def attention_pass(model_path: str, samples: list, device: str = "cuda"):
    """For each sample: generate w/ output_attentions=True, extract per-layer attn-to-image."""
    print(f"\n[attn] Loading {Path(model_path).name} (eager)")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device,
        attn_implementation="eager", trust_remote_code=True,
    ).eval()

    results = []
    for i, s in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] id={s['id']}", flush=True)
        pil = Image.open(s["image"]).convert("RGB")
        inputs = build_inputs_for_generation(processor, pil, s["question"], device)
        input_ids = inputs.input_ids[0]
        image_positions = (input_ids == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]

        thw = inputs["image_grid_thw"][0].tolist()
        merge = getattr(model.config.vision_config, "spatial_merge_size", 2)
        gh, gw = thw[1] // merge, thw[2] // merge

        out = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            output_attentions=True, return_dict_in_generate=True, use_cache=True,
        )
        per_layer_vec = extract_per_layer_attn_to_image(out, image_positions, LAST_N_LAYERS)
        per_layer_grid = {li: v.reshape(gh, gw) for li, v in per_layer_vec.items()}

        # also keep generated text for reference
        gen_ids = out.sequences[0, input_ids.shape[0]:]
        gen_text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)

        results.append({
            "id": s["id"], "image": s["image"], "question": s["question"],
            "gt_bboxes": s["gt_bboxes"], "gt_label": s["gt_label"],
            "gen_text": gen_text, "pred_label": parse_final_answer(gen_text),
            "per_layer": per_layer_grid, "image_size": pil.size,
        })

        del out, inputs
        torch.cuda.empty_cache()

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return results


def upsample_heatmap(heatmap: np.ndarray, size_wh: tuple) -> np.ndarray:
    W, H = size_wh
    img = Image.fromarray(heatmap.astype(np.float32)).resize((W, H), resample=Image.BILINEAR)
    return np.asarray(img)


def draw_bboxes(ax, bboxes: list):
    for x1, y1, x2, y2 in bboxes:
        ax.add_patch(mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="lime", facecolor="none",
        ))


def render_overlay(ax, pil_image: Image.Image, heatmap_grid: np.ndarray, bboxes: list, title: str):
    hm = upsample_heatmap(heatmap_grid, pil_image.size).astype(np.float32)
    if hm.max() > hm.min():
        hm = (hm - hm.min()) / (hm.max() - hm.min())
    ax.imshow(pil_image)
    ax.imshow(hm, cmap="jet", alpha=0.45)
    draw_bboxes(ax, bboxes)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def render_per_sample(sft_attn: list, cbm_attn: list, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for sft, cbm in zip(sft_attn, cbm_attn):
        pil = Image.open(sft["image"]).convert("RGB")
        sft_layers = sorted(sft["per_layer"].keys())  # ascending: e.g. [24,25,26,27]
        cbm_layers = sorted(cbm["per_layer"].keys())
        common = [l for l in sft_layers if l in cbm["per_layer"]]
        K = len(common)
        if K == 0:
            continue
        fig, axes = plt.subplots(K, 3, figsize=(15, 5 * K))
        if K == 1:
            axes = axes[None, :]
        for r, li in enumerate(common):
            axes[r, 0].imshow(pil); draw_bboxes(axes[r, 0], sft["gt_bboxes"]); axes[r, 0].axis("off")
            axes[r, 0].set_title(f"layer L={li}  (GT bboxes lime)", fontsize=10)
            render_overlay(axes[r, 1], pil, sft["per_layer"][li], sft["gt_bboxes"],
                           f"cbm-only (sft baseline) layer {li}")
            render_overlay(axes[r, 2], pil, cbm["per_layer"][li], sft["gt_bboxes"],
                           f"cbm+contrastive layer {li}")
        q_short = (sft["question"][:140] + "…") if len(sft["question"]) > 140 else sft["question"]
        fig.suptitle(
            f"id={sft['id']}  GT={sft['gt_label']}  |  cbm-only pred={sft['pred_label']}  ✗  cbm+contrastive pred={cbm['pred_label']}  ✓\nQ: {q_short}",
            fontsize=11,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out_path = out_dir / f"sample_{sft['id']}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_path}  ({K} layers)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_json", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--baseline_model", required=True, help="cbm-only model (the 'sft' baseline)")
    ap.add_argument("--target_model", required=True, help="cbm+contrastive model")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pool_size", type=int, default=100)
    ap.add_argument("--num_viz", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = load_pool(args.test_json, args.image_dir, args.pool_size, args.seed)
    print(f"[main] pool size = {len(pool)} (seed={args.seed})")

    # Pass 1: predictions
    print("\n[main] === Pass 1: predictions for baseline (cbm-only) ===")
    base_pred = predict_pass(args.baseline_model, pool)
    print("\n[main] === Pass 1: predictions for cbm+contrastive ===")
    targ_pred = predict_pass(args.target_model, pool)

    by_id_b = {r["id"]: r for r in base_pred}
    by_id_t = {r["id"]: r for r in targ_pred}

    n_b = sum(1 for r in base_pred if r["correct"])
    n_t = sum(1 for r in targ_pred if r["correct"])
    print(f"\n[main] baseline (cbm-only)        accuracy: {n_b}/{len(pool)}")
    print(f"[main] target   (cbm+contrastive) accuracy: {n_t}/{len(pool)}")

    # Filter: target correct AND baseline wrong
    interesting = [s for s in pool
                   if by_id_t[s["id"]]["correct"] and not by_id_b[s["id"]]["correct"]]
    print(f"[main] samples where cbm+contrastive correct AND cbm-only wrong: {len(interesting)}")

    if not interesting:
        # save predictions before bailing
        (out_dir / "_predictions.json").write_text(json.dumps({
            "baseline": base_pred, "target": targ_pred,
        }, indent=2))
        print("[main] no interesting samples found — exiting after dumping predictions.")
        return

    picked = interesting[: args.num_viz]
    print(f"[main] picking first {len(picked)} for attention extraction:")
    for s in picked:
        print(f"  id={s['id']:>6}  gt={s['gt_label']}  base={by_id_b[s['id']]['pred_label']}  targ={by_id_t[s['id']]['pred_label']}")

    # Pass 2: attentions on filtered
    print("\n[main] === Pass 2: attention extraction (baseline) ===")
    base_attn = attention_pass(args.baseline_model, picked)
    print("\n[main] === Pass 2: attention extraction (target) ===")
    targ_attn = attention_pass(args.target_model, picked)

    print("\n[main] === Rendering ===")
    render_per_sample(base_attn, targ_attn, out_dir)

    summary = []
    for s in picked:
        b, t = by_id_b[s["id"]], by_id_t[s["id"]]
        summary.append({
            "id": s["id"], "gt_label": s["gt_label"],
            "baseline_pred": b["pred_label"], "target_pred": t["pred_label"],
            "baseline_gen": b["gen_text"], "target_gen": t["gen_text"],
            "image": s["image"], "question": s["question"],
        })
    (out_dir / "_predictions.json").write_text(json.dumps({
        "baseline_acc": f"{n_b}/{len(pool)}", "target_acc": f"{n_t}/{len(pool)}",
        "n_target_correct_base_wrong": len(interesting),
        "picked": summary,
        "all_baseline": base_pred, "all_target": targ_pred,
    }, indent=2))
    print(f"\n[main] Done. Output: {out_dir}")


if __name__ == "__main__":
    main()

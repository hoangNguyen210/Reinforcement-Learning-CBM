"""
Compare per-CONCEPT image attention of two Qwen2-VL checkpoints (SFT vs
SFT+CBM+Contrastive) using TEACHER-FORCED GT answers.

Why this approach
-----------------
The CBM model has identical architecture and weight-tensor set as SFT — the
concept supervision was a training-time auxiliary loss only, so there is no
"concept layer" to read. To still test concept grounding, we put the GT chain-
of-thought (which contains concept words like 'sulci', 'atrophy', 'cingulate')
into the assistant turn and look at attention from those concept tokens to the
image-patch tokens. If CBM training actually grounded concepts to image regions,
attention from those tokens should land on the GT bbox more often than for SFT.

For each sample:
  1. Build prompt = system + user(image, question) + assistant(GT_answer).
  2. Single forward pass with output_attentions=True (eager attention).
  3. Find the assistant token span; for each KEY_TERM that appears in it, find
     the token positions; take attention from those positions -> image patches,
     averaged over heads + last 4 LLM layers + the matched-token positions;
     reshape to the patch grid; upsample; score = sum(attn in bbox)/sum(attn).
  4. Sample-level score = mean of per-concept scores for that sample.

Pick top NUM_SAMPLES (8) by Δ = cbm_score - sft_score. For each picked sample,
render one PNG with K rows (one per matched concept) × 3 columns
(original | SFT heatmap | CBM heatmap), GT bboxes overlaid in green.
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

IMAGE_TOKEN_ID = 151655      # <|image_pad|>
VISION_END_TOKEN_ID = 151653  # <|vision_end|>
IM_START_TOKEN_ID = 151644   # <|im_start|>
IM_END_TOKEN_ID = 151645     # <|im_end|>
LAST_N_LAYERS = 4
MAX_CONCEPTS_RENDERED = 5  # cap rows per sample figure
BBOX_RE = re.compile(r"```json\s*(\[.*?\])\s*```", re.S)

# Anatomical / finding terms that appear in the GT CoT answers. Ordered by
# frequency in the test set (verified against llava_med_mri_bbox_test_CoT_new).
# Skipping score-tag terms like 'Koedam','GCA','MTA' (not anatomical).
KEY_TERMS = [
    "atrophy", "sulci", "cingulate", "occipital", "parieto-occipital",
    "posterior", "widening", "cortical", "ventricular", "temporal", "ventricle",
    "parietal", "enlarged", "sulcal", "gyral", "hippocampus", "frontal",
    "sulcus", "gyrus", "precuneus", "white matter", "lesion",
]


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
    have_bbox = [d for d in data if parse_gt_bboxes(d["conversations"][1]["value"])]
    picked = rng.sample(have_bbox, pool_size)
    out = []
    for item in picked:
        question = item["conversations"][0]["value"].replace("<image>", "").strip()
        img_path = Path(image_dir) / item["image"]
        bboxes = parse_gt_bboxes(item["conversations"][1]["value"])
        gt_answer = item["conversations"][1]["value"]
        out.append({
            "id": item["id"], "image": str(img_path),
            "question": question, "gt_bboxes": bboxes, "gt_answer": gt_answer,
        })
    return out


def build_inputs_teacher_forced(processor, pil_image: Image.Image, question: str,
                                 gt_answer: str, device):
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": question},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": gt_answer}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = processor(text=[text], images=[pil_image], return_tensors="pt").to(device)
    return inputs


def find_assistant_span(input_ids: torch.Tensor):
    """Return (start, end) token indices for the assistant content, exclusive
    of the role marker tokens (<|im_start|>, 'assistant', '\n') and the closing
    <|im_end|>."""
    ids = input_ids.cpu()
    starts = (ids == IM_START_TOKEN_ID).nonzero(as_tuple=True)[0]
    if len(starts) == 0:
        raise RuntimeError("no <|im_start|> token in prompt")
    last_start = int(starts[-1])
    ends = (ids == IM_END_TOKEN_ID).nonzero(as_tuple=True)[0]
    after = ends[ends > last_start]
    if len(after) == 0:
        raise RuntimeError("no <|im_end|> after assistant <|im_start|>")
    end = int(after[0])
    # The role marker is "<|im_start|>assistant\n" → 3 tokens. Skip them.
    span_start = last_start + 3
    if span_start >= end:
        raise RuntimeError("empty assistant span")
    return span_start, end


def find_term_token_positions(tokenizer, input_ids: torch.Tensor, term: str,
                                span_start: int, span_end: int) -> list:
    """Find all token positions in input_ids[span_start:span_end] that
    correspond to occurrences of `term`. Tries both with leading space and
    without (BPE often produces different tokens)."""
    span = input_ids[span_start:span_end].cpu().tolist()
    found_positions = []
    seen = set()
    for lead in (" ", ""):
        toks = tokenizer(lead + term, add_special_tokens=False)["input_ids"]
        if not toks:
            continue
        n = len(toks)
        for i in range(len(span) - n + 1):
            if span[i:i + n] == toks:
                start = span_start + i
                if start in seen:
                    continue
                seen.add(start)
                found_positions.extend(range(start, start + n))
    return sorted(set(found_positions))


def precompute_attn_to_image(outputs, image_positions: torch.Tensor,
                              last_n_layers: int) -> list:
    """Slice attn[layer][0, :, :, image_positions] on GPU, then move once to
    CPU per layer. Returns list of (H, S, n_image) float32 CPU tensors. Doing
    this once per forward pass (instead of once per concept) keeps GPU↔CPU
    transfer cost ~constant rather than scaling with K_concepts × layers."""
    layers = outputs.attentions[-last_n_layers:]
    out = []
    for attn in layers:
        a = attn[0][:, :, image_positions]   # (H, S, n_image) on GPU
        out.append(a.detach().cpu().float())
    return out


def extract_per_concept_attn(per_layer_attn_to_image: list,
                              src_positions: torch.Tensor) -> np.ndarray:
    """Slice the src dimension on CPU (cheap), avg over heads + src + layers."""
    src = src_positions.cpu()
    n_img = per_layer_attn_to_image[0].shape[-1]
    accum = torch.zeros(n_img, dtype=torch.float32)
    for a in per_layer_attn_to_image:
        accum += a[:, src, :].mean(dim=(0, 1))   # (n_img,)
    return (accum / len(per_layer_attn_to_image)).numpy()


def upsample_heatmap(heatmap: np.ndarray, size_wh: tuple) -> np.ndarray:
    W, H = size_wh
    img = Image.fromarray(heatmap.astype(np.float32)).resize((W, H), resample=Image.BILINEAR)
    return np.asarray(img)


def attention_in_bbox_score(heatmap_full: np.ndarray, bboxes: list) -> float:
    if not bboxes:
        return float("nan")
    H, W = heatmap_full.shape
    mask = np.zeros((H, W), dtype=bool)
    for x1, y1, x2, y2 in bboxes:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(W, int(x2)), min(H, int(y2))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
    total = float(heatmap_full.sum())
    if total <= 0:
        return float("nan")
    return float(heatmap_full[mask].sum() / total)


@torch.inference_mode()
def run_model_on_samples(model_path: str, samples: list, device: str = "cuda"):
    print(f"\n[viz] Loading processor from {model_path}")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    print(f"[viz] Loading model {Path(model_path).name} (eager attn)")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.eval()

    results = []
    for i, s in enumerate(samples):
        print(f"  [{i + 1}/{len(samples)}] id={s['id']}  image={Path(s['image']).name}", flush=True)
        pil = Image.open(s["image"]).convert("RGB")
        inputs = build_inputs_teacher_forced(processor, pil, s["question"], s["gt_answer"], device)
        input_ids = inputs.input_ids[0]
        image_positions = (input_ids == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]

        thw = inputs["image_grid_thw"][0].tolist()
        merge = getattr(model.config.vision_config, "spatial_merge_size", 2)
        gh, gw = thw[1] // merge, thw[2] // merge
        assert gh * gw == image_positions.numel()

        try:
            span_start, span_end = find_assistant_span(input_ids)
        except RuntimeError as e:
            print(f"    skip: {e}")
            results.append({"id": s["id"], "concepts": {}, "score_mean": float("nan")})
            continue

        # which key terms appear in the assistant span (token-level match)
        per_term_positions = {}
        for term in KEY_TERMS:
            pos = find_term_token_positions(tokenizer, input_ids, term, span_start, span_end)
            if pos:
                per_term_positions[term] = pos
        if not per_term_positions:
            print("    skip: no key term matched in assistant span")
            results.append({"id": s["id"], "concepts": {}, "score_mean": float("nan")})
            continue

        outputs = model(**inputs, output_attentions=True, use_cache=False)
        per_layer_attn_to_image = precompute_attn_to_image(outputs, image_positions, LAST_N_LAYERS)
        del outputs
        torch.cuda.empty_cache()

        per_concept = {}
        for term, positions in per_term_positions.items():
            src = torch.tensor(positions, dtype=torch.long)
            attn_vec = extract_per_concept_attn(per_layer_attn_to_image, src)
            heatmap = attn_vec.reshape(gh, gw)
            heatmap_full = upsample_heatmap(heatmap, pil.size)
            score = attention_in_bbox_score(heatmap_full, s["gt_bboxes"])
            per_concept[term] = {"heatmap_full": heatmap_full, "score": score, "n_tokens": len(positions)}
        del per_layer_attn_to_image

        scores = [c["score"] for c in per_concept.values() if c["score"] == c["score"]]
        score_mean = float(np.mean(scores)) if scores else float("nan")

        results.append({
            "id": s["id"], "image": s["image"], "question": s["question"],
            "gt_bboxes": s["gt_bboxes"], "image_size": pil.size,
            "concepts": per_concept, "score_mean": score_mean,
        })

        del inputs
        torch.cuda.empty_cache()

    del model, processor, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return results


def draw_bboxes(ax, bboxes: list):
    for x1, y1, x2, y2 in bboxes:
        ax.add_patch(mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="lime", facecolor="none",
        ))


def render_overlay(ax, pil_image: Image.Image, heatmap_full: np.ndarray, bboxes: list, title: str):
    hm = heatmap_full.astype(np.float32)
    if hm.max() > hm.min():
        hm = (hm - hm.min()) / (hm.max() - hm.min())
    ax.imshow(pil_image)
    ax.imshow(hm, cmap="jet", alpha=0.45)
    draw_bboxes(ax, bboxes)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def render_per_sample(picked: list, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in picked:
        sft, cbm = row["sft"], row["cbm"]
        # only render concepts present in BOTH model results
        common_terms = sorted(set(sft["concepts"]) & set(cbm["concepts"]),
                              key=lambda t: -(cbm["concepts"][t]["score"] - sft["concepts"][t]["score"]))
        common_terms = common_terms[:MAX_CONCEPTS_RENDERED]
        if not common_terms:
            continue
        pil = Image.open(sft["image"]).convert("RGB")
        K = len(common_terms)
        fig, axes = plt.subplots(K, 3, figsize=(15, 5 * K))
        if K == 1:
            axes = axes[None, :]
        for r, term in enumerate(common_terms):
            sft_c = sft["concepts"][term]
            cbm_c = cbm["concepts"][term]
            d = cbm_c["score"] - sft_c["score"]
            axes[r, 0].imshow(pil); draw_bboxes(axes[r, 0], sft["gt_bboxes"]); axes[r, 0].axis("off")
            axes[r, 0].set_title(f"concept: '{term}'  Δ={d:+.3f}", fontsize=10)
            render_overlay(axes[r, 1], pil, sft_c["heatmap_full"], sft["gt_bboxes"],
                           f"SFT  ({sft_c['score']:.3f})")
            render_overlay(axes[r, 2], pil, cbm_c["heatmap_full"], cbm["gt_bboxes"],
                           f"SFT+CBM+Contrastive  ({cbm_c['score']:.3f})")
        q_short = (sft["question"][:140] + "…") if len(sft["question"]) > 140 else sft["question"]
        fig.suptitle(f"id={sft['id']}  Q: {q_short}\nsample-mean Δ = {cbm['score_mean']-sft['score_mean']:+.3f}",
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out_path = out_dir / f"sample_{sft['id']}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_path}  ({K} concepts)")


def dump_pool_scores(sft_res, cbm_res, picked_ids: set, out_path: Path):
    rows = []
    by_id_cbm = {r["id"]: r for r in cbm_res}
    for sft in sft_res:
        cbm = by_id_cbm[sft["id"]]
        common = sorted(set(sft["concepts"]) & set(cbm["concepts"]))
        per_term = {}
        for t in common:
            per_term[t] = {
                "sft_score": sft["concepts"][t]["score"],
                "cbm_score": cbm["concepts"][t]["score"],
                "delta": cbm["concepts"][t]["score"] - sft["concepts"][t]["score"],
            }
        rows.append({
            "id": sft["id"],
            "sft_score_mean": sft["score_mean"], "cbm_score_mean": cbm["score_mean"],
            "delta_mean": cbm["score_mean"] - sft["score_mean"],
            "per_concept": per_term,
            "picked": sft["id"] in picked_ids,
        })
    rows.sort(key=lambda r: -(r["delta_mean"] if r["delta_mean"] == r["delta_mean"] else -1))
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"  saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_json", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--sft_model", required=True)
    ap.add_argument("--cbm_model", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pool_size", type=int, default=60)
    ap.add_argument("--num_samples", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = load_pool(args.test_json, args.image_dir, args.pool_size, args.seed)
    print(f"[viz] pool size = {len(pool)} (seed={args.seed}); top {args.num_samples} by Δ(cbm−sft) per-concept-mean")

    print("\n[viz] === Running SFT model on pool ===")
    sft_res = run_model_on_samples(args.sft_model, pool)
    print("\n[viz] === Running SFT+CBM+Contrastive model on pool ===")
    cbm_res = run_model_on_samples(args.cbm_model, pool)

    by_id_sft = {r["id"]: r for r in sft_res}
    by_id_cbm = {r["id"]: r for r in cbm_res}
    rows = []
    for sid in by_id_sft:
        s, c = by_id_sft[sid], by_id_cbm[sid]
        d = c["score_mean"] - s["score_mean"]
        if d == d:
            rows.append({"id": sid, "delta": d, "sft": s, "cbm": c})
    rows.sort(key=lambda r: -r["delta"])
    n_pos = sum(1 for r in rows if r["delta"] > 0)
    print(f"\n[viz] {n_pos}/{len(rows)} samples have CBM > SFT on per-concept-mean attn-in-bbox")
    picked = rows[: args.num_samples]
    print(f"\n[viz] Top {len(picked)} samples by Δ:")
    for p in picked:
        print(f"  id={p['id']:>6}  sft={p['sft']['score_mean']:.3f}  cbm={p['cbm']['score_mean']:.3f}  Δ={p['delta']:+.3f}")

    print("\n[viz] === Rendering ===")
    render_per_sample(picked, out_dir)
    dump_pool_scores(sft_res, cbm_res, {p["id"] for p in picked}, out_dir / "_scores.json")
    print(f"\n[viz] Done. Output: {out_dir}")


if __name__ == "__main__":
    main()

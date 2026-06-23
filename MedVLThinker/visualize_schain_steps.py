"""
Visualize per-step attention for S-Chain Qwen2-VL models.

For each sample produces a figure with 5 panels:
  Col 0: Original MRI + GT bbox (lime) overlaid
  Col 1: Full reasoning span  — all tokens inside + Reasoning:…Final Answer → image patches
  Col 2: Step 1 answer tokens — bbox localization → image patches
  Col 3: Step 2 answer tokens — concept description → image patches
  Col 4: Step 3 answer tokens — grade/class → image patches

With --model_b, renders 2 rows per sample (model A top, model B bottom).
Each heatmap also shows the per-step attention-in-GT-bbox score.

Usage — single model:
  python3 visualize_schain_steps.py \\
      --test_json  /path/to/llava_med_mri_bbox_test_CoT_new.json \\
      --image_dir  /path/to/images \\
      --model_a    /path/to/checkpoint \\
      --out_dir    figs/schain_steps_sft40 \\
      [--model_a_label "SFT 40% ep9"] \\
      [--num_samples 8] [--pool_size 60] [--seed 42] [--device cuda]

Two-model comparison (SFT vs GRPO):
  python3 visualize_schain_steps.py \\
      --test_json  ... --image_dir ... \\
      --model_a    /path/to/sft_ckpt  --model_a_label "SFT 40% ep9" \\
      --model_b    /path/to/grpo_ckpt --model_b_label "GRPO 60% ep3" \\
      --out_dir    figs/schain_steps_sft_vs_grpo \\
      [--sort_by step1]   # pick samples with largest Δ on Step 1 attn-in-bbox
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

# ── Token ID constants for Qwen2-VL ──────────────────────────────────────────
IMAGE_TOKEN_ID    = 151655   # <|image_pad|>
IM_START_TOKEN_ID = 151644   # <|im_start|>
IM_END_TOKEN_ID   = 151645   # <|im_end|>

LAST_N_LAYERS = 4

STEP_LABELS = {
    "think":  "Q1 — Full reasoning",
    1:        "Q2 — Step 1 (bbox)",
    2:        "Q3 — Step 2 (concepts)",
    3:        "Q4 — Step 3 (grade/class)",
}

BBOX_RE = re.compile(r'"bbox_2d"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]')
ANSWER_RE  = re.compile(r'-\s*Answer\s*:', re.IGNORECASE)
STEP_RE    = re.compile(r'-\s*Step\s+(\d+)\s*:', re.IGNORECASE)
FINAL_RE   = re.compile(r'\+\s*Final\s+Answer\s*:', re.IGNORECASE)
THINK_END  = re.compile(r'</think>', re.IGNORECASE)


# ── Data loading ──────────────────────────────────────────────────────────────

def parse_gt_bboxes(gt_text: str) -> list:
    return [[int(a), int(b), int(c), int(d)] for a, b, c, d in BBOX_RE.findall(gt_text)]


def load_pool(test_json: str, image_dir: str, pool_size: int, seed: int) -> list:
    """Random pool filtered to samples that have GT bboxes (for standalone use)."""
    with open(test_json) as f:
        data = json.load(f)
    rng = random.Random(seed)
    have_bbox = [d for d in data if parse_gt_bboxes(d["conversations"][1]["value"])]
    picked = rng.sample(have_bbox, min(pool_size, len(have_bbox)))
    return _build_records(picked, image_dir)


def load_pool_qualitative(test_json: str, image_dir: str, num_samples: int,
                           seed: int) -> list:
    """Exact same sampling as qualitative_grpo_compare.py so the same 5 images
    are used. No bbox filter — mirrors the qualitative script precisely."""
    with open(test_json) as f:
        data = json.load(f)
    random.seed(seed)
    indices = random.sample(range(len(data)), min(num_samples, len(data)))
    picked  = [data[i] for i in indices]
    return _build_records(picked, image_dir)


def _build_records(items: list, image_dir: str) -> list:
    out = []
    for item in items:
        question  = item["conversations"][0]["value"].replace("<image>", "").strip()
        gt_answer = item["conversations"][1]["value"]
        img_path  = Path(image_dir) / item["image"]
        bboxes    = parse_gt_bboxes(gt_answer)
        out.append({
            "id": item["id"], "image": str(img_path),
            "question": question, "gt_answer": gt_answer, "gt_bboxes": bboxes,
        })
    return out


# ── Input construction ────────────────────────────────────────────────────────

def build_inputs_teacher_forced(processor, pil_image: Image.Image,
                                 question: str, gt_answer: str, device: str):
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": question},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": gt_answer}]},
    ]
    text   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = processor(text=[text], images=[pil_image], return_tensors="pt").to(device)
    return inputs


# ── Assistant span detection ──────────────────────────────────────────────────

def find_assistant_span(input_ids: torch.Tensor):
    """Return (start, end) of the assistant content, skipping role marker tokens."""
    ids     = input_ids.cpu()
    starts  = (ids == IM_START_TOKEN_ID).nonzero(as_tuple=True)[0]
    if len(starts) == 0:
        raise RuntimeError("no <|im_start|> in input_ids")
    last_start = int(starts[-1])
    ends = (ids == IM_END_TOKEN_ID).nonzero(as_tuple=True)[0]
    after = ends[ends > last_start]
    if len(after) == 0:
        raise RuntimeError("no <|im_end|> after assistant start")
    span_end   = int(after[0])
    span_start = last_start + 3   # skip <|im_start|> + "assistant" + "\n" tokens
    if span_start >= span_end:
        raise RuntimeError("empty assistant span")
    return span_start, span_end


# ── Step span extraction ──────────────────────────────────────────────────────

def _decode_span_with_charmap(tokenizer, tok_ids: list) -> tuple:
    """
    Decode a list of token ids one-by-one and build a mapping
    char_idx → position_within_tok_ids. Returns (decoded_text, char_to_tok).
    """
    parts, char_to_tok = [], []
    for tok_idx, tok_id in enumerate(tok_ids):
        piece = tokenizer.decode([tok_id], skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)
        parts.append(piece)
        char_to_tok.extend([tok_idx] * len(piece))
    return "".join(parts), char_to_tok


def _char_range_to_abs_tokens(char_to_tok: list, char_start: int,
                               char_end: int, span_start: int) -> list:
    """Convert a character range into a sorted list of absolute token indices."""
    char_end = min(char_end, len(char_to_tok))
    if char_start >= char_end:
        return []
    return sorted(set(span_start + char_to_tok[i]
                      for i in range(char_start, char_end)))


def find_step_spans(tokenizer, input_ids: torch.Tensor,
                    span_start: int, span_end: int) -> dict:
    """
    Parse the assistant span and return token positions for each step section.

    Returns dict with keys 'think', 1, 2, 3 — each a sorted list of
    absolute token indices within input_ids.

    Layout expected in the decoded text:
        + Reasoning:
        - Step 1: ...
        - Answer: <step-1 content>
        - Step 2: ...
        - Answer: <step-2 content>
        - Step 3: ...
        - Answer: <step-3 content>
        + Final Answer: ...
    """
    tok_ids = input_ids[span_start:span_end].cpu().tolist()
    text, c2t = _decode_span_with_charmap(tokenizer, tok_ids)

    # Find boundaries
    answer_ends   = [m.end()   for m in ANSWER_RE.finditer(text)]   # after "- Answer:"
    step_starts   = [m.start() for m in STEP_RE.finditer(text)]     # start of "- Step N:"
    final_start   = (m := FINAL_RE.search(text)) and m.start() or len(text)
    think_end_pos = (m := THINK_END.search(text)) and m.start() or len(text)

    # Full reasoning span: start of text to end of think block (or Final Answer)
    reasoning_end = min(think_end_pos, len(text))

    spans = {}

    spans["think"] = _char_range_to_abs_tokens(c2t, 0, reasoning_end, span_start)

    # Step N answer span: from end of Nth "- Answer:" to start of next "- Step:" or final
    for i, ans_char_start in enumerate(answer_ends[:3]):
        step_num = i + 1
        # end of this answer = start of next step marker (or final answer or think_end)
        candidates = [s for s in step_starts if s > ans_char_start]
        next_step  = candidates[0] if candidates else final_start
        ans_char_end = min(next_step, final_start, think_end_pos)
        spans[step_num] = _char_range_to_abs_tokens(
            c2t, ans_char_start, ans_char_end, span_start)

    return spans


# ── Attention extraction ──────────────────────────────────────────────────────

def precompute_attn_to_image(outputs, image_positions: torch.Tensor,
                              last_n_layers: int) -> list:
    """Slice attn[layer][:, :, image_pos] on GPU, move to CPU. Returns list of
    (H, S, n_img) float32 CPU tensors — one per layer."""
    layers = outputs.attentions[-last_n_layers:]
    result = []
    for attn in layers:
        a = attn[0][:, :, image_positions]   # (H, S, n_img)
        result.append(a.detach().cpu().float())
    return result


def extract_span_attn(per_layer_attn: list, src_positions: list) -> np.ndarray:
    """Average attention from src_positions → image patches over heads + layers."""
    if not src_positions:
        n_img = per_layer_attn[0].shape[-1]
        return np.zeros(n_img, dtype=np.float32)
    src = torch.tensor(src_positions, dtype=torch.long)
    n_img = per_layer_attn[0].shape[-1]
    accum = torch.zeros(n_img, dtype=torch.float32)
    for a in per_layer_attn:
        # a: (H, S, n_img) — slice src dim on CPU
        accum += a[:, src, :].mean(dim=(0, 1))
    return (accum / len(per_layer_attn)).numpy()


def upsample_heatmap(heatmap: np.ndarray, size_wh: tuple) -> np.ndarray:
    W, H = size_wh
    img = Image.fromarray(heatmap.astype(np.float32)).resize((W, H), resample=Image.BILINEAR)
    return np.asarray(img)


def attn_in_bbox_score(heatmap: np.ndarray, bboxes: list) -> float:
    if not bboxes:
        return float("nan")
    H, W = heatmap.shape
    mask = np.zeros((H, W), dtype=bool)
    for x1, y1, x2, y2 in bboxes:
        mask[max(0, int(y1)):min(H, int(y2)),
             max(0, int(x1)):min(W, int(x2))] = True
    total = float(heatmap.sum())
    return float(heatmap[mask].sum() / total) if total > 0 else float("nan")


# ── Main inference loop ───────────────────────────────────────────────────────

@torch.inference_mode()
def run_model(model_path: str, samples: list, device: str = "cuda", processor_path: str = "") -> list:
    print(f"\n[viz] Loading {Path(model_path).name} (eager attn)")
    proc_path = processor_path or model_path
    processor = AutoProcessor.from_pretrained(proc_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
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
        print(f"  [{i+1}/{len(samples)}] id={s['id']}  {Path(s['image']).name}", flush=True)
        pil    = Image.open(s["image"]).convert("RGB")
        inputs = build_inputs_teacher_forced(processor, pil, s["question"], s["gt_answer"], device)
        ids    = inputs.input_ids[0]

        img_pos = (ids == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
        thw   = inputs["image_grid_thw"][0].tolist()
        merge = getattr(model.config.vision_config, "spatial_merge_size", 2)
        gh, gw = thw[1] // merge, thw[2] // merge
        assert gh * gw == img_pos.numel(), \
            f"grid mismatch: {gh}×{gw}={gh*gw} vs {img_pos.numel()} image tokens"

        try:
            span_start, span_end = find_assistant_span(ids)
        except RuntimeError as e:
            print(f"    skip: {e}")
            results.append({"id": s["id"], "skip": True})
            continue

        step_spans = find_step_spans(tokenizer, ids, span_start, span_end)

        empty = {k: [] for k in step_spans if not step_spans[k]}
        if empty:
            print(f"    warn: empty spans for {list(empty)}")

        outputs = model(**inputs, output_attentions=True, use_cache=False)
        per_layer = precompute_attn_to_image(outputs, img_pos, LAST_N_LAYERS)
        del outputs
        torch.cuda.empty_cache()

        per_step = {}
        for key, tok_positions in step_spans.items():
            attn_vec  = extract_span_attn(per_layer, tok_positions)
            heatmap   = attn_vec.reshape(gh, gw)
            heatmap_f = upsample_heatmap(heatmap, pil.size)
            score     = attn_in_bbox_score(heatmap_f, s["gt_bboxes"])
            per_step[key] = {
                "heatmap": heatmap_f,
                "score":   score,
                "n_tokens": len(tok_positions),
            }

        del per_layer, inputs
        torch.cuda.empty_cache()

        results.append({
            "id":        s["id"],
            "image":     s["image"],
            "image_size": pil.size,
            "gt_bboxes": s["gt_bboxes"],
            "question":  s["question"],
            "per_step":  per_step,
            "skip":      False,
        })

    del model, processor, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return results


# ── Rendering ─────────────────────────────────────────────────────────────────

def draw_bboxes(ax, bboxes: list, color: str = "lime", lw: int = 2):
    for x1, y1, x2, y2 in bboxes:
        ax.add_patch(mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=lw, edgecolor=color, facecolor="none",
        ))


def render_heatmap(ax, pil: Image.Image, heatmap: np.ndarray,
                   bboxes: list, title: str):
    hm = heatmap.astype(np.float32)
    if hm.max() > hm.min():
        hm = (hm - hm.min()) / (hm.max() - hm.min())
    ax.imshow(pil)
    ax.imshow(hm, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    draw_bboxes(ax, bboxes)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


STEP_KEYS_ALL = ["think", 1, 2, 3]   # column order (after original image)
STEP_KEYS = STEP_KEYS_ALL            # overridden by --num_steps at runtime


def render_sample(record_a, label_a: str,
                  record_b=None, label_b: str = "",
                  out_path: Path = None):
    n_rows = 2 if record_b is not None else 1
    n_cols = 1 + len(STEP_KEYS)          # original + 4 step heatmaps
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4.5 * n_cols, 4.5 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]              # (1, n_cols)

    for row_idx, (record, label) in enumerate(
            [(record_a, label_a)] + ([(record_b, label_b)] if record_b else [])):
        pil = Image.open(record["image"]).convert("RGB")

        # Col 0: original
        axes[row_idx, 0].imshow(pil)
        draw_bboxes(axes[row_idx, 0], record["gt_bboxes"])
        axes[row_idx, 0].set_title(f"{label}\n(GT bbox)", fontsize=9)
        axes[row_idx, 0].axis("off")

        # Cols 1‥4: step heatmaps
        for col_idx, key in enumerate(STEP_KEYS, start=1):
            ax    = axes[row_idx, col_idx]
            info  = record["per_step"].get(key)
            col_title = STEP_LABELS[key]
            if info is None or info["n_tokens"] == 0:
                ax.imshow(pil); ax.axis("off")
                ax.set_title(f"{col_title}\n(no tokens)", fontsize=9)
                continue
            score_str = (f"{info['score']:.3f}"
                          if info["score"] == info["score"] else "—")
            render_heatmap(ax, pil, info["heatmap"], record["gt_bboxes"],
                           f"{col_title}\nattn-in-bbox={score_str}")

    id_str = record_a["id"]
    fig.suptitle(f"id={id_str}", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ── Score JSON ────────────────────────────────────────────────────────────────

def _safe(v):
    return None if (isinstance(v, float) and v != v) else v


def dump_scores(res_a, label_a: str, res_b, label_b: str,
                picked_ids: set, out_path: Path):
    by_id_b = {r["id"]: r for r in res_b} if res_b else {}
    rows = []
    for ra in res_a:
        if ra.get("skip"):
            continue
        row = {"id": ra["id"], "models": {}}
        row["models"][label_a] = {
            k: _safe(ra["per_step"][k]["score"])
            for k in STEP_KEYS if k in ra["per_step"]
        }
        if ra["id"] in by_id_b:
            rb = by_id_b[ra["id"]]
            if not rb.get("skip"):
                row["models"][label_b] = {
                    k: _safe(rb["per_step"][k]["score"])
                    for k in STEP_KEYS if k in rb["per_step"]
                }
                row["delta_step1"] = _safe(
                    (rb["per_step"].get(1, {}).get("score", float("nan")) or float("nan")) -
                    (ra["per_step"].get(1, {}).get("score", float("nan")) or float("nan"))
                )
        row["picked"] = ra["id"] in picked_ids
        rows.append(row)
    if res_b:
        rows.sort(key=lambda r: -(r.get("delta_step1") or -1))
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"  saved scores → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_json",     required=True)
    ap.add_argument("--image_dir",     required=True)
    ap.add_argument("--model_a",       required=True,
                    help="Primary checkpoint (SFT or GRPO)")
    ap.add_argument("--model_b",       default="",
                    help="Optional second checkpoint for comparison")
    ap.add_argument("--model_a_label", default="Model A")
    ap.add_argument("--model_b_label", default="Model B")
    ap.add_argument("--processor",     default="",
                    help="Path to processor/tokenizer (defaults to model_a path)")
    ap.add_argument("--out_dir",       required=True)
    ap.add_argument("--match_qualitative", action="store_true",
                    help="Use the exact same 5-sample selection as qualitative_grpo_compare.py "
                         "(same seed + random.sample logic, no bbox filter). "
                         "Overrides --pool_size and --num_samples.")
    ap.add_argument("--pool_size",     type=int, default=60,
                    help="Random pool to sample from before picking top-N")
    ap.add_argument("--num_samples",   type=int, default=8,
                    help="Number of sample figures to render")
    ap.add_argument("--sort_by",       default="step1",
                    choices=["step1", "think", "random"],
                    help="How to rank samples (only matters with 2 models: delta on that key)")
    ap.add_argument("--num_steps",     type=int, default=3,
                    help="Number of reasoning steps (2 for LoBa, 3 for S-Chain)")
    ap.add_argument("--seed",          type=int, default=42)
    ap.add_argument("--device",        default="cuda")
    args = ap.parse_args()

    global STEP_KEYS
    STEP_KEYS = ["think"] + list(range(1, args.num_steps + 1))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.match_qualitative:
        pool = load_pool_qualitative(args.test_json, args.image_dir,
                                     num_samples=5, seed=args.seed)
        print(f"[viz] match_qualitative: {len(pool)} samples (seed={args.seed}, same as qualitative_grpo_compare.py)")
    else:
        pool = load_pool(args.test_json, args.image_dir, args.pool_size, args.seed)
        print(f"[viz] pool={len(pool)} samples  (seed={args.seed})")

    proc_path = args.processor or args.model_a
    res_a = run_model(args.model_a, pool, args.device, processor_path=proc_path)

    res_b = None
    if args.model_b:
        res_b = run_model(args.model_b, pool, args.device, processor_path=proc_path)

    # ── pick top samples ──────────────────────────────────────────────────────
    valid_a = {r["id"]: r for r in res_a if not r.get("skip")}
    if args.match_qualitative:
        # Pool IS the target set — render every valid sample
        picked_ids = list(valid_a.keys())
        valid_b = {r["id"]: r for r in res_b if not r.get("skip")} if res_b else {}
    elif res_b:
        valid_b = {r["id"]: r for r in res_b if not r.get("skip")}
        shared  = [sid for sid in valid_a if sid in valid_b]

        sort_key_map = {"step1": 1, "think": "think"}
        sk = sort_key_map.get(args.sort_by, 1)

        def delta(sid):
            sa = valid_a[sid]["per_step"].get(sk, {}).get("score", float("nan"))
            sb = valid_b[sid]["per_step"].get(sk, {}).get("score", float("nan"))
            if sa != sa or sb != sb:
                return float("-inf")
            return sb - sa

        ranked = sorted(shared, key=delta, reverse=True)
        picked_ids = ranked[: args.num_samples]
        print(f"\n[viz] Top {len(picked_ids)} by Δ({args.sort_by})  [B − A]:")
        for sid in picked_ids:
            d = delta(sid)
            sa = valid_a[sid]["per_step"].get(1, {}).get("score", float("nan"))
            sb = valid_b[sid]["per_step"].get(1, {}).get("score", float("nan"))
            print(f"  id={sid:>6}  A={sa:.3f}  B={sb:.3f}  Δ={d:+.3f}")
    else:
        rng = random.Random(args.seed)
        picked_ids = rng.sample(list(valid_a), min(args.num_samples, len(valid_a)))
        valid_b = {}

    # ── render ────────────────────────────────────────────────────────────────
    print(f"\n[viz] Rendering {len(picked_ids)} figures → {out_dir}")
    for sid in picked_ids:
        ra = valid_a[sid]
        rb = valid_b.get(sid) if res_b else None
        render_sample(
            ra, args.model_a_label,
            rb, args.model_b_label,
            out_path=out_dir / f"sample_{sid}.png",
        )

    dump_scores(res_a, args.model_a_label, res_b, args.model_b_label,
                set(picked_ids), out_dir / "_scores.json")
    print(f"\n[viz] Done → {out_dir}")


if __name__ == "__main__":
    main()

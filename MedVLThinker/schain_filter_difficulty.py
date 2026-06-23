"""
schain_filter_difficulty.py — Pass-rate difficulty filter for S-Chain training data.

Adapts MedVLThinker's curriculum-learning idea to the S-Chain dataset.
Supports --num_shards / --shard_idx for data-parallel runs across multiple GPUs.

Typical call (one shard per GPU, launched in parallel by run_schain_filter_parallel.sh):
    CUDA_VISIBLE_DEVICES=3 python schain_filter_difficulty.py \\
        --input  ablation1_40percent_all_questions.json \\
        --images /path/to/images \\
        --save_pass_counts /tmp/40pct_shard3.jsonl \\
        --shard_idx 3 --num_shards 8

Then collect all shard JSONLs and run with --merge_shards to produce the filtered JSON:
    python schain_filter_difficulty.py --merge_shards \\
        --input  ablation1_40percent_all_questions.json \\
        --images /path/to/images \\
        --output filtered_40pct.json \\
        --shard_pattern /tmp/40pct_shard*.jsonl \\
        --min_pass 1 --max_pass 6
"""

import argparse
import glob
import json
import os
import re
from collections import Counter
from math import ceil
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

CLASSES = ["Non-Dementia", "Mild-Dementia", "Moderate-Dementia"]
CLASS_RE = re.compile(r"(Non-Dementia|Mild-Dementia|Moderate-Dementia)", re.IGNORECASE)
FINAL_RE = re.compile(
    r"Final\s*Answer\s*[:.]?\s*(Non-Dementia|Mild-Dementia|Moderate-Dementia)",
    re.IGNORECASE,
)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

SYSTEM_PROMPT = (
    "You are an expert radiologist specialising in dementia MRI analysis. "
    "Analyse the scan step-by-step and always end your response with exactly "
    "one of: Non-Dementia, Mild-Dementia, or Moderate-Dementia."
)

MODEL_DEFAULT = (
    "/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/SChain/model_weights/Qwen3-VL-8B-Instruct"
)


# ── GT / prediction helpers ───────────────────────────────────────────────────

def extract_gt_class(gpt_value: str) -> str:
    m = FINAL_RE.search(gpt_value)
    if m:
        return m.group(1)
    stripped = gpt_value.strip()
    for c in CLASSES:
        if stripped.lower() == c.lower():
            return c
    return ""


def score_pred(pred: str, gt_class: str) -> bool:
    pred = THINK_RE.sub("", pred)
    m = FINAL_RE.search(pred)
    if m:
        return m.group(1).lower() == gt_class.lower()
    m = CLASS_RE.search(pred[-300:])
    if m:
        return m.group(1).lower() == gt_class.lower()
    return False


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(model_path: str):
    print(f"[shard] Loading processor from {model_path} ...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print(f"[shard] Loading model (bfloat16, device_map=auto) ...")
    # AutoModelForImageTextToText covers all VLMs: Qwen3-VL, Qwen2.5-VL, LLaVA, etc.
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return processor, model


def run_rollouts(processor, model, question: str, pil_image: Image.Image,
                 n_rollouts: int, temperature: float, max_new_tokens: int) -> list[str]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": (
                f"{question}\n\n"
                "Think step-by-step. End your answer with exactly one of: "
                "Non-Dementia, Mild-Dementia, or Moderate-Dementia."
            )},
        ]},
    ]
    try:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    inputs = processor(text=[text], images=[pil_image], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=n_rollouts,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    new_tokens = out_ids[:, input_len:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)


# ── Load + validate dataset ───────────────────────────────────────────────────

def load_valid(input_json: str, images_dir: str):
    with open(input_json) as f:
        records = json.load(f)
    valid = []
    for rec in records:
        img_path = os.path.join(images_dir, rec["image"])
        if not os.path.exists(img_path):
            continue
        convs = rec.get("conversations", [])
        if len(convs) < 2:
            continue
        gt = extract_gt_class(convs[1]["value"])
        if not gt:
            continue
        question = convs[0]["value"].replace("<image>", "").strip()
        valid.append((rec, img_path, gt, question))
    return records, valid


# ── Inference shard ───────────────────────────────────────────────────────────

def run_shard(args):
    records, valid = load_valid(args.input, args.images)
    print(f"[shard {args.shard_idx}/{args.num_shards}] "
          f"Total valid: {len(valid)} / {len(records)}")

    # Contiguous slice for this shard
    chunk = ceil(len(valid) / args.num_shards)
    start = args.shard_idx * chunk
    end = min(start + chunk, len(valid))
    my_samples = valid[start:end]
    print(f"[shard {args.shard_idx}] Processing samples [{start}, {end}) = {len(my_samples)} samples")

    processor, model = load_model(args.model)

    pass_counts = []
    for rec, img_path, gt, question in tqdm(my_samples,
                                             desc=f"shard-{args.shard_idx}"):
        pil_image = Image.open(img_path).convert("RGB")
        preds = run_rollouts(
            processor, model, question, pil_image,
            args.n_rollouts, args.temperature, args.max_new_tokens,
        )
        n_correct = sum(score_pred(p, gt) for p in preds)
        pass_counts.append({
            "id": rec["id"],
            "gt_class": gt,
            "pass_count": n_correct,
            "n_rollouts": args.n_rollouts,
        })

    Path(args.save_pass_counts).parent.mkdir(parents=True, exist_ok=True)
    with open(args.save_pass_counts, "w") as f:
        for pc in pass_counts:
            f.write(json.dumps(pc) + "\n")
    print(f"[shard {args.shard_idx}] Saved {len(pass_counts)} pass counts → {args.save_pass_counts}")


# ── Merge shards + filter ─────────────────────────────────────────────────────

def run_merge(args):
    shard_files = sorted(glob.glob(args.shard_pattern))
    if not shard_files:
        raise FileNotFoundError(f"No shard files found: {args.shard_pattern}")
    print(f"Merging {len(shard_files)} shard files ...")

    pass_map = {}
    for sf in shard_files:
        with open(sf) as f:
            for line in f:
                if line.strip():
                    pc = json.loads(line)
                    pass_map[pc["id"]] = pc["pass_count"]

    # Distribution report
    records, valid = load_valid(args.input, args.images)
    dist = Counter(pass_map.get(rec["id"], -1) for rec, _, _, _ in valid)
    n = args.n_rollouts
    print(f"\nPass-count distribution (N={n} rollouts per sample):")
    for k in sorted(dist):
        tag = ("KEEP" if args.min_pass <= k <= args.max_pass
               else ("NOT SCORED" if k == -1
               else ("TOO HARD" if k < args.min_pass else "TOO EASY")))
        pct = dist[k] / len(valid) * 100
        print(f"  pass={k:2d}: {dist[k]:5d}  ({pct:5.1f}%)  [{tag}]")

    # Filter
    filtered = [
        rec for rec, _, _, _ in valid
        if args.min_pass <= pass_map.get(rec["id"], -1) <= args.max_pass
    ]
    print(f"\nKept {len(filtered)} / {len(valid)} samples "
          f"(pass_count in [{args.min_pass}, {args.max_pass}])")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)
    print(f"Saved → {args.output}")
    print("Drop this file into --data_path / DATA_PATH in your training scripts.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--input",    required=True, help="Training JSON")
    ap.add_argument("--images",   required=True, help="Image root directory")
    ap.add_argument("--model",    default=MODEL_DEFAULT,
                    help="Model path. Swap to any VLM (Qwen2.5-VL, etc.) here.")

    # Shard mode
    ap.add_argument("--shard_idx",  type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--save_pass_counts", type=str, default=None,
                    help="JSONL to write this shard's pass counts")

    # Merge mode
    ap.add_argument("--merge_shards",  action="store_true",
                    help="Merge shard JSONLs and write filtered JSON")
    ap.add_argument("--shard_pattern", type=str, default=None,
                    help="Glob pattern for shard JSONL files (merge mode)")
    ap.add_argument("--output",  type=str, default=None,
                    help="Filtered output JSON (merge mode)")

    # Filter thresholds
    ap.add_argument("--min_pass",    type=int, default=1)
    ap.add_argument("--max_pass",    type=int, default=6)

    # Inference settings
    ap.add_argument("--n_rollouts",     type=int,   default=8)
    ap.add_argument("--temperature",    type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int,   default=256)

    args = ap.parse_args()

    if args.merge_shards:
        if not args.output:
            ap.error("--output is required in --merge_shards mode")
        if not args.shard_pattern:
            ap.error("--shard_pattern is required in --merge_shards mode")
        run_merge(args)
    else:
        if not args.save_pass_counts:
            ap.error("--save_pass_counts is required in shard mode")
        run_shard(args)


if __name__ == "__main__":
    main()

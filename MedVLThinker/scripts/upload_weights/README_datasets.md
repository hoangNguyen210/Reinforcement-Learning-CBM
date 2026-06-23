# Dataset Upload Script

This script uploads datasets from the `med-vlrm` organization to the `UCSC-VLAA` organization on Hugging Face.

## Dataset Mappings

The script uploads the following datasets:

- `med-vlrm/med-vlm-pmc_vqa-gpt_4o_reasoning-tokenized` → `UCSC-VLAA/MedVLThinker-pmc_vqa-gpt_4o_reasoning-tokenized`
- `med-vlrm/med-vlm-eval-v2` → `UCSC-VLAA/MedVLThinker-Eval`
- `med-vlrm/med-vlm-pmc_vqa` → `UCSC-VLAA/MedVLThinker-pmc_vqa`
- `med-vlrm/med-vlm-m23k-tokenized` → `UCSC-VLAA/MedVLThinker-m23k-tokenized`

## Prerequisites

1. Install required packages:
   ```bash
   pip install -r requirements_datasets.txt
   ```

2. Install git-lfs (for handling large files):
   ```bash
   # On Ubuntu/Debian
   sudo apt install git-lfs
   
   # On macOS
   brew install git-lfs
   
   # Initialize git-lfs
   git lfs install
   ```

3. Authenticate with Hugging Face:
   ```bash
   huggingface-cli login
   ```
   Or provide your token directly to the script.

## Usage

### Upload all datasets:
```bash
python upload_datasets.py
```

### Dry run (see what would be done without actually uploading):
```bash
python upload_datasets.py --dry-run
```

### Upload a specific dataset:
```bash
python upload_datasets.py --dataset med-vlrm/med-vlm-eval-v2
```

### Provide token directly:
```bash
python upload_datasets.py --token YOUR_HF_TOKEN
```

## Options

- `--token`: Hugging Face API token (optional if already logged in)
- `--dry-run`: Show what would be done without actually uploading
- `--dataset`: Upload only a specific dataset (source repo ID)

## Notes

- The script will automatically create target repositories if they don't exist
- Large files are handled via git-lfs
- Progress and errors are logged to the console
- The script provides a summary at the end showing successful and failed uploads

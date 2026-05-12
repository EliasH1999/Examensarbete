# Assisted Configuration Generation Based on Device Manuals

This repository contains the full implementation used in our thesis project. Given a device manual, the system extracts configuration parameters and outputs a structured configuration in a fixed schema (key=value)

## What's inside
We provide multiple extraction methods for comparison
- **Prompt-only (3-tiers)**
    - `naive_user`: open-ended questions (no UI constraints)
    - `configurator_user`: configurator-aware questions but constrained to valid UI options
    - `engineeered_prompt`: engineered prompt with Modbus rules
**Naive RAG**: chunk -> embed -> top-k retrieval -> single-pass generation
**Modular RAG**: multiple targeted retrieval/generation passes + ranking + validation + merge

We provide evaluation utilities:
- `run_benchmark.py` - run methods over a set of manuals and save predictions
- `compare.py` - compares predictions to golden keyvaleus
- `accuracy.py` - computes accuracy metrics

## Repository files

- **Run / benchmark**
  - `run_benchmark.py` — main benchmark runner
  - `random_manual_generator.py` — was used for randomizing ten manuals used for prompt engineering
  - `pred.txt` — example output file (can be overwritten)
- **Pipelines**
  - `basic_prompt.py` — prompt-only baseline helper(s)
  - `naive_user.py` — prompt-only tier 1
  - `configurator_user.py` — prompt-only tier 2
  - `engineered_prompt.py` — prompt-only tier 3
  - `naive_rag.py` — naive RAG pipeline
  - `modular_rag.py` — modular RAG pipeline
  **Evaluation**
  - `compare.py`, `checkkey.py`, `accuracy.py`
- **Utilities**
  - `fc16_echo.py` — helper for FC16/write payload cases
  - `functioncode_changer` — address/function-code normalization helper
  - `changes.py` — misc changes/utilities
  - `make_golden_keyValues.py` — generate/format golden keyvalue files

## Dataset (manuals + golden configs)

PDF manuals are **not redistributed** in this repository.

Our dataset repository contains:
- `manual_inventory.csv` (manual_id, source_url, access_date, golden_file)
- `golden_keyValues/` (ground truth key=value files)

## Dataset
Pdf manuals are not redistributed in this repository or in the dataset repository

The accompanying dataset repository contains the manually curated reference data used for evaluation:
- `manual_inventory.csv` with `manual_id`, `source_url`, `access_date`, and `golden_file`
- `golden_keyValues` containing the ground-truth `key=value` files
 
The original manuals must be obtained from their respective source URLs listed in `manual_inventory.csv`
Dataset repo: https://github.com/EliasH1999/dataset_contribution

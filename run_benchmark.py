"""
Automated Benchmark Runner
===========================
Runs extraction on all (or random subset of) manuals, compares predictions
against golden keyValues, and writes results to a timestamped CSV.

CONFIGURATION
-------------
Edit the two variables below to control behavior:

  MANUALS = "random"   ->  only the 10 randomly selected manuals
  MANUALS = "all"      ->  all 52 manuals with golden keyValues

  METHOD  = "prompt"       ->  prompt-based extraction  (prompt.py)
  METHOD  = "naive_rag"    ->  naive RAG extraction     (naive_rag.py)
  METHOD  = "modular_rag"  ->  modular RAG extraction   (modular_rag.py)

The AI model is configured inside each extraction script (e.g. prompt.py),
not here. Change it there if you want a different model.

USAGE
-----
  python run_benchmark.py
"""

import json
import csv
import os
import re
import sys
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

from accuracy import compute_accuracy, readable_param

# CONFIGURATION - edit these variables to change behavior
# MANUALS: "random" = only the 10 random manuals
#          "all"    = all 52 manuals
MANUALS = "all"

# METHOD: which extraction script to run (see METHOD_SCRIPTS below)
METHOD = "modular_rag"

# Final five thesis methods.
METHOD_SCRIPTS = {
    "naive_user": "naive_user.py",
    "configurator_user": "configurator_user.py",
    "engineered_prompt": "engineered_prompt.py",
    "naive_rag": "naive_rag.py",
    "modular_rag": "modular_rag.py",
}

# Base directories (relative to this script)
BASE_DIR = Path(__file__).resolve().parent
MAPPING_FILE = BASE_DIR / "mapping.json"
GOLDEN_DIR = BASE_DIR / "golden_keyValues"
ALL_MANUALS_DIR = BASE_DIR / "all_manuals"
RANDOM_MANUALS_DIR = BASE_DIR / "random_manuals"
PREDICTIONS_DIR = BASE_DIR / "predictions"
RESULTS_DIR = BASE_DIR / "results"  # Results CSVs go in results/{method}/
CHUNK_DB_DIR = BASE_DIR / "chunk_db"  # Saved chunks for post-run inspection


def load_mapping():
    """Load and optionally filter the mapping.json entries.

    Optional CLI filter: pass one or more substrings to run only matching manuals.
        python run_benchmark.py 1815243
        python run_benchmark.py 1815243 APM_MAX
    The filter overrides the MANUALS setting and matches against the manual name
    or pdf filename (case-insensitive substring match).
    """
    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        entries = json.load(f)

    cli_filters = [a.lower() for a in sys.argv[1:] if not a.startswith("-")]
    if cli_filters:
        def _matches(e):
            hay = (str(e.get("name", "")) + " " + str(e.get("pdf", ""))).lower()
            return any(f in hay for f in cli_filters)
        entries = [e for e in entries if _matches(e)]
        if not entries:
            print(f"No manuals matched filter(s): {cli_filters}")
            sys.exit(1)
        print(f"CLI filter active: running {len(entries)} manual(s) matching {cli_filters}")
        return entries

    if MANUALS == "random":
        entries = [e for e in entries if e.get("random", False)]

    return entries


def extract_register_name(golden_kv_path: str) -> str:
    """Read a golden keyValues file and extract the register/transaction name."""
    with open(golden_kv_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line.startswith("subnetwork.nodes[0].transactions[0].properties.name="):
                return line.split("=", 1)[1]
    return "unknown"


def build_question(register_name: str) -> str:
    """Build a standardized question from the register name."""
    # Replace underscores with spaces (improves RAG embedding similarity).
    # "Power_factor_A" won't match PDF text "Power factor", but "Power factor A" will.
    extracted_name = register_name.replace("_", " ")
    if METHOD in (
        "naive_user",
        "configurator_user",
        "engineered_prompt",
        "naive_rag",
        "modular_rag",
    ):
        return f'I want the "{extracted_name}" register and all the communication settings'
    return (
        f'give me communication settings, nodes and transactions settings '
        f'for the slave and for the "{extracted_name}" register'
    )


def resolve_pdf_path(entry: dict) -> Path:
    """Resolve the PDF path - prefer random_manuals/ if in random mode and available."""
    pdf_name = entry["pdf"]
    if MANUALS == "random" and entry.get("random", False):
        random_path = RANDOM_MANUALS_DIR / pdf_name
        if random_path.exists():
            return random_path
    return ALL_MANUALS_DIR / pdf_name


def detect_model_name(script_path: Path) -> str:
    """Parse the extraction script source to find the model name."""
    try:
        source = script_path.read_text(encoding="utf-8")
        # Look for init_chat_model("model_name", ...)
        match = re.search(r'init_chat_model\(["\']([^"\']+)["\']', source)
        if match:
            return match.group(1)
    except Exception:
        pass
    return "unknown"


def clean_pdf_name(dirty_name: str) -> str:
    """Create a clean display name from a dirty PDF filename.
    
    Strips trailing dots/spaces and double .pdf extensions for display in CSV.
    """
    name = dirty_name.strip()
    # Remove trailing junk like "....pdf" or ".pdf     .pdf"
    # First strip trailing whitespace and dots before the final .pdf
    while True:
        # Remove patterns like ".pdf     .pdf" -> ".pdf"
        new_name = re.sub(r'\.pdf\s+\.pdf$', '.pdf', name, flags=re.IGNORECASE)
        # Remove trailing dots and spaces before .pdf
        new_name = re.sub(r'[\s.]+\.pdf$', '.pdf', new_name, flags=re.IGNORECASE)
        if new_name == name:
            break
        name = new_name
    
    # If it doesn't end with .pdf, add it
    if not name.lower().endswith('.pdf'):
        name += '.pdf'
    
    return name


def run_extraction(pdf_path: Path, question: str, output_path: Path):
    """Call the extraction script as a subprocess."""
    script = METHOD_SCRIPTS.get(METHOD)
    if script is None:
        raise ValueError(f"Unknown method: {METHOD}. Valid: {list(METHOD_SCRIPTS.keys())}")

    script_path = BASE_DIR / script
    if not script_path.exists():
        raise FileNotFoundError(f"Extraction script not found: {script_path}")

    cmd = [
        sys.executable, script,
        "--manual",   str(pdf_path),
        "--question",  question,
        "--output",   str(output_path),
    ]

    result = subprocess.run(
        cmd,
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        timeout=300,  # 5 minute timeout per manual
    )

    if result.returncode != 0:
        print(f"  WARNING: Extraction returned non-zero exit code {result.returncode}")
        if result.stderr:
            print(f"  STDERR: {result.stderr[:2000]}")

    # Show debug output from extraction script
    if result.stdout:
        for line in result.stdout.splitlines():
            if line.startswith("[DEBUG]") or line.startswith("  Chunk"):
                print(f"         {line}")

    return result


def write_csv_row(writer, manual_name: str, model_name: str,
                  correct: int, total: int,
                  pct: float, results: list):
    """Write accuracy results for one manual to the CSV.

    Layout per manual:
      - 1 summary row: manual name + accuracy
      - N parameter rows (one per golden field): status + parameter, predicted
        Sorted: Correct -> Non-default match -> Wrong -> Excluded
    """
    from accuracy import readable_param, STATUS_LABEL, _format_predicted_cell

    writer.writerow({
        "manual": manual_name,
        "ai_model": model_name,
        "accuracy": f"{correct}/{total}",
        "accuracy_percent": f"{pct:.2f}%",
    })

    for row in results:
        label = STATUS_LABEL[row["status"]]
        exp_line = f"{row['key']}={row['default']}"
        writer.writerow({
            "manual": "",
            "ai_model": "",
            "accuracy": f"{label}: {readable_param(exp_line)}",
            "accuracy_percent": _format_predicted_cell(row),
        })


def main():
    # Load mapping
    entries = load_mapping()
    if not entries:
        print(f"No manuals found for MANUALS={MANUALS}. Check mapping.json.")
        sys.exit(1)

    # Detect model name from extraction script
    script_path = BASE_DIR / METHOD_SCRIPTS[METHOD]
    model_name = detect_model_name(script_path)

    # Clear chunk_db from previous run so it only contains current results
    if CHUNK_DB_DIR.exists():
        try:
            shutil.rmtree(CHUNK_DB_DIR)
        except PermissionError:
            # Files may be open in VS Code - delete contents individually
            for f in CHUNK_DB_DIR.iterdir():
                try:
                    f.unlink()
                except PermissionError:
                    pass  # skip locked files
    CHUNK_DB_DIR.mkdir(exist_ok=True)

    # Create predictions directory
    PREDICTIONS_DIR.mkdir(exist_ok=True)

    # Create timestamped output CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    method_results_dir = RESULTS_DIR / METHOD
    method_results_dir.mkdir(parents=True, exist_ok=True)
    csv_name = f"results_{METHOD}_{timestamp}.csv"
    csv_path = method_results_dir / csv_name

    print(f"=" * 60)
    print(f"Benchmark Runner")
    print(f"  Manuals:   {MANUALS} ({len(entries)} manuals)")
    print(f"  Method:    {METHOD} ({METHOD_SCRIPTS[METHOD]})")
    print(f"  Model:     {model_name}")
    print(f"  Output:    {csv_name}")
    print(f"=" * 60)
    print()

    accuracies = []

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "manual", "ai_model",
            "accuracy", "accuracy_percent",
        ])
        writer.writeheader()

        for i, entry in enumerate(entries, 1):
            pdf_name = entry["pdf"]
            golden_kv_name = entry["golden_keyValues"]
            display_name = clean_pdf_name(pdf_name)

            # Resolve paths
            pdf_path = resolve_pdf_path(entry)
            golden_kv_path = GOLDEN_DIR / golden_kv_name

            # Validate files exist
            if not pdf_path.exists():
                print(f"[{i}/{len(entries)}] SKIP {display_name} - PDF not found: {pdf_path}")
                continue
            if not golden_kv_path.exists():
                print(f"[{i}/{len(entries)}] SKIP {display_name} - Golden keyValues not found: {golden_kv_path}")
                continue

            # Extract register name and build question
            register_name = extract_register_name(str(golden_kv_path))
            question = build_question(register_name)

            # Prediction output file
            safe_stem = re.sub(r'[^\w\-.]', '_', Path(pdf_name).stem)
            pred_path = PREDICTIONS_DIR / f"{safe_stem}_pred.txt"

            print(f"[{i}/{len(entries)}] {display_name}")
            print(f"         Register: {register_name}")
            print(f"         Question: {question}")

            # Delete stale prediction file so we never reuse old results
            if pred_path.exists():
                pred_path.unlink()

            # Run extraction
            try:
                run_extraction(pdf_path, question, pred_path)
            except subprocess.TimeoutExpired:
                print(f"         TIMEOUT - skipping")
                continue
            except Exception as e:
                print(f"         ERROR: {e} - skipping")
                continue

            # Check if prediction file was created
            if not pred_path.exists():
                print(f"         No prediction file generated - skipping")
                continue

            # Compute accuracy
            correct, total, pct, results = compute_accuracy(
                str(golden_kv_path), str(pred_path)
            )

            accuracies.append(pct)
            print(f"         Accuracy: {pct:.2f}% ({correct}/{total})  [Node Name + Transaction Name excluded]")
            from accuracy import PARAMETER_NAMES, STATUS_LABEL
            for row in results:
                if row["status"] == "correct" and not row["note"]:
                    continue  # don't spam terminal with default-correct rows
                name = PARAMETER_NAMES.get(row["key"], row["key"])
                label = STATUS_LABEL[row["status"]]
                pred_display = row["predicted"] if row["predicted"] is not None else "NOT FOUND"
                suffix = ""
                if row["status"] == "non_default":
                    suffix = f"  ({row['note']})"
                elif row["status"] == "wrong":
                    suffix = f"  (expected={row['default']})"
                elif row["status"] == "excluded":
                    suffix = f"  ({row['note']})"
                elif row["status"] == "correct" and row["note"]:
                    suffix = f"  ({row['note']})"
                print(f"           [{label}] {name}: predicted={pred_display}{suffix}")
            print()

            # Write to CSV
            write_csv_row(writer, display_name, model_name,
                          correct, total, pct, results)

        # Write average row at the end of the CSV
        if accuracies:
            avg = sum(accuracies) / len(accuracies)
            writer.writerow({
                "manual": "AVERAGE",
                "ai_model": model_name,
                "accuracy": f"{len(accuracies)} manuals",
                "accuracy_percent": f"{avg:.2f}%",
            })

    # Summary
    print(f"=" * 60)
    print(f"SUMMARY")
    print(f"  Manuals evaluated: {len(accuracies)}/{len(entries)}")
    if accuracies:
        avg = sum(accuracies) / len(accuracies)
        print(f"  Avg accuracy:      {avg:.2f}%  (min {min(accuracies):.2f}%, max {max(accuracies):.2f}%)")
        print(f"  (Node Name + Transaction Name excluded from scoring)")
    print(f"  Results saved to:  {csv_name}")
    print(f"=" * 60)


if __name__ == "__main__":
    main()

"""
Token-usage validation driver.

Runs all five extraction methods against a small, size-stratified subset of
manuals (p10 / p25 / p50 / p75 / p90 by PDF size) and lets each script append
its real OpenAI usage numbers to results/token_usage.csv via usage_logger.py.

Run:
    python validate_token_usage.py
"""

import json
import re
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
PREDICTIONS = BASE / "predictions" / "_validation"
PREDICTIONS.mkdir(parents=True, exist_ok=True)

METHODS = {
    "naive_user":        "naive_user.py",
    "configurator_user": "configurator_user.py",
    "engineered_prompt": "engineered_prompt.py",
    "naive_rag":         "naive_rag.py",
    "modular_rag":       "modular_rag.py",
}

# Size-stratified sample (chosen to span PDF token range without burning $$
# on the 10 MB outlier).
SUBSET_PDFS = [
    "s5gbm_modbus-installation-kep-x.pdf",                             # p10
    "Instruction_Manual_VA_5xx_Modbus_RTU_Slave_Installation.pdf",     # p25
    "man--thermoMETER-ct-ctlaser-modbus-rtu-commands--en.pdf",         # p50
    "Modbus-RTU-Manual.2022.pdf",                                      # p75
    "PM5100.pdf",                                                      # p90
]


def resolve_pdf(pdf_name):
    for sub in ("all_manuals", "random_manuals"):
        p = BASE / sub / pdf_name
        if p.exists():
            return p
    raise FileNotFoundError(pdf_name)


def extract_register_name(golden_path):
    with open(golden_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line.startswith("subnetwork.nodes[0].transactions[0].properties.name="):
                return line.split("=", 1)[1]
    return "unknown"


def main():
    mapping = json.loads((BASE / "mapping.json").read_text(encoding="utf-8"))
    by_pdf = {e["pdf"]: e for e in mapping}

    print(f"Validation subset: {len(SUBSET_PDFS)} manuals - {len(METHODS)} methods "
          f"= {len(SUBSET_PDFS) * len(METHODS)} runs (modular_rag = 2 LLM calls each)")
    print("=" * 70)

    for pdf_name in SUBSET_PDFS:
        if pdf_name not in by_pdf:
            print(f"SKIP {pdf_name}: not in mapping.json")
            continue
        entry = by_pdf[pdf_name]
        pdf_path = resolve_pdf(pdf_name)
        golden_path = BASE / "golden_keyValues" / entry["golden_keyValues"]
        register = extract_register_name(golden_path).replace("_", " ")
        question = f'I want the "{register}" register and all the communication settings'
        safe_stem = re.sub(r"[^\w\-.]", "_", Path(pdf_name).stem)

        print(f"\n[{pdf_name}]  register='{register}'")
        for method, script in METHODS.items():
            out = PREDICTIONS / f"{safe_stem}__{method}.txt"
            if out.exists():
                out.unlink()
            cmd = [
                sys.executable, script,
                "--manual",   str(pdf_path),
                "--question", question,
                "--output",   str(out),
            ]
            print(f"  -> {method:22s} ", end="", flush=True)
            r = subprocess.run(cmd, cwd=str(BASE), capture_output=True,
                               text=True, timeout=600)
            status = "OK" if r.returncode == 0 else f"EXIT {r.returncode}"
            print(status)
            if r.returncode != 0 and r.stderr:
                print(r.stderr.strip().splitlines()[-3:])

    print("\nDone. See results/token_usage.csv for the appended rows.")


if __name__ == "__main__":
    main()

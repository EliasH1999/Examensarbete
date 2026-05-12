import random
import secrets
import sys
from pathlib import Path
from datetime import datetime, timezone

# Folder with all manuals
folder = Path(r"C:\Users\william\OneDrive - Högskolan i Halmstad\UNIVERSITET\VT26 Examensarbete\GITHUB\Promtbased_test\examensarbete_test\all_manuals")

# Folder to save the output
output_folder = Path(r"C:\Users\william\OneDrive - Högskolan i Halmstad\UNIVERSITET\VT26 Examensarbete\GITHUB\Promtbased_test\examensarbete_test\random_manuals")
output_folder.mkdir(parents=True, exist_ok=True)

# Safety check - stop if the file already exists
output_file = output_folder / "random_manual_generator_sample.txt"
if output_file.exists():
    sys.exit("ERROR: random_manual_generator_sample.txt already exists. This script should only be run once.")

# Get all PDFs and sort them
pdfs = sorted(folder.glob("*.pdf"))
total = len(pdfs)

# Use OS-level true randomness (not algorithm-based)
# This pulls randomness from the operating system (/dev/urandom or CryptGenRandom)
secure_rng = random.SystemRandom()

# Generate a seed and record it so the run can be verified later
seed = secrets.randbits(64)

# Pick 10 random numbers between 1 and total number of manuals
picks = sorted(secure_rng.sample(range(1, total + 1), 10))

# Build the output text
lines = []
lines.append("RANDOM MANUAL SELECTION - SIMPLE RANDOM SAMPLE")
lines.append(f"Date/time (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
lines.append(f"Seed:            {seed}")
lines.append(f"Total manuals:   {total}")
lines.append(f"Range:           1 to {total}")
lines.append(f"Selected:        10")
lines.append(f"Method:          Simple random sample without replacement")
lines.append(f"Randomness:      OS-level (random.SystemRandom)")
lines.append("")
lines.append("This script was designed to only run once.")
lines.append("The existence of this file proves the selection was not re-rolled.")

lines.append("\nALL MANUALS:\n")
for i, pdf in enumerate(pdfs, 1):
    lines.append(f"  [{i}] {pdf.name}")

lines.append(f"\nRANDOM PICKS: {picks}\n")
for num in picks:
    lines.append(f"  #{num} -> {pdfs[num - 1].name}")

lines.append("")

# Join everything into one string
output = "\n".join(lines)

# Print to screen
print(output)

# Save to file
output_file.write_text(output, encoding="utf-8")
print(f"\nSaved to: {output_file}")
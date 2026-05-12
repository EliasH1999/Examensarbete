import argparse
import csv
from pathlib import Path


# Human-readable names for the configuration parameters.
# Maps the config-path prefix (before '=') to a readable label.
PARAMETER_NAMES = {
    "subnetwork.properties.physicalStandard":                                          "Physical Standard",
    "subnetwork.properties.baudRate":                                                  "Baud Rate",
    "subnetwork.properties.parity":                                                    "Parity",
    "subnetwork.properties.stopBits":                                                  "Stop Bits",
    "subnetwork.properties.dataBits":                                                  "Data Bits",
    "subnetwork.nodes[0].properties.nodeAddress":                                      "Node Address",
    "subnetwork.nodes[0].properties.name":                                             "Node Name",
    "subnetwork.nodes[0].properties.modbusAddressingMode":                             "Addressing Mode",
    "subnetwork.nodes[0].transactions[0].properties.name":                             "Transaction Name",
    "subnetwork.nodes[0].transactions[0].frames[0].frameObjects[1].properties.data":   "Request Function Code",
    "subnetwork.nodes[0].transactions[0].frames[0].frameObjects[2].properties.data":   "Request Starting Address",
    "subnetwork.nodes[0].transactions[0].frames[0].frameObjects[3].properties.data":   "Request Quantity",
    "subnetwork.nodes[0].transactions[0].frames[1].frameObjects[1].properties.data":   "Response Function Code",
    "subnetwork.nodes[0].transactions[0].frames[1].frameObjects[2].properties.data":   "Response Byte Count",
    "subnetwork.nodes[0].transactions[0].frames[1].frameObjects[3].properties.dataLength": "Response Data Length",
}

# Keys excluded from accuracy scoring.
# These are free-text naming-convention fields (Node Name, Transaction Name)
# that do not represent extracted facts from the manual - they are house-style
# labels. They are still shown in the wrong_pairs report for debugging, but
# do not contribute to the numerator or denominator of the accuracy metric.
EXCLUDED_KEYS = {
    "subnetwork.nodes[0].properties.name",
    "subnetwork.nodes[0].transactions[0].properties.name",
}


def readable_param(line: str) -> str:
    """Convert a key=value line to 'Readable Name (key=value)' format."""
    key = line.split("=", 1)[0] if "=" in line else line
    name = PARAMETER_NAMES.get(key)
    if name:
        return f"{name} ({line})"
    return line


def read_set(path: str):
    with open(path, "r", encoding="utf-8-sig") as f:
        return {line.strip() for line in f if line.strip()}


# PDF-row equivalence
# Keys involved in the coupled (addressing_mode, starting_address) pair.
# Two predictions are considered equivalent if they reference the same
# original PDF row number, regardless of which addressing mode was chosen.
_KEY_ADDR_MODE = "subnetwork.nodes[0].properties.modbusAddressingMode"
_KEY_START_ADDR = "subnetwork.nodes[0].transactions[0].frames[0].frameObjects[2].properties.data"


def _to_wire_addresses(mode_str, addr_str):
    """Convert (addressingMode, startingAddress) to a SET containing the wire address.

    Returns an empty set if the address is missing/unparseable.

    The system prompt instructs the LLM to pre-apply the addressing-mode
    transformation when filling startingAddress (mode=1 -> register-1,
    mode=2 -> modicon-40001, etc.). As a result, the stored value already
    represents the wire address regardless of which mode was chosen, and
    the downstream configurator consumes only the address. Therefore two
    predictions are equivalent iff their startingAddress values match,
    independent of the addressing mode.
    """
    try:
        addr = int(addr_str)
    except (TypeError, ValueError):
        return set()
    return {addr}


def _parse_golden_values(raw: str):
    """Split a golden value string on '|' into (default, alternates).

    Format:
        '9600'                  -> default='9600',  alternates=[]
        '9600|2400|4800|19200'  -> default='9600',  alternates=['2400','4800','19200']
        '247|1-246'             -> default='247',   alternates=['1-246']  (range)

    The first value is the documented factory default (preferred answer).
    Remaining values are also-acceptable values; each can be:
      - a single literal value (e.g. '9600')
      - an inclusive integer range 'N-M' (e.g. '1-246')
    """
    parts = [p.strip() for p in raw.split("|") if p.strip()]
    if not parts:
        return "", []
    return parts[0], parts[1:]


def _matches_alternate(predicted, alternates):
    """Return True if predicted matches any alternate (literal or N-M range)."""
    if predicted is None:
        return False
    for alt in alternates:
        if alt == predicted:
            return True
        # Try integer range 'N-M' (inclusive)
        if "-" in alt:
            lo_s, hi_s = alt.split("-", 1)
            try:
                lo = int(lo_s.strip())
                hi = int(hi_s.strip())
                p_int = int(predicted)
            except (TypeError, ValueError):
                continue
            if lo <= p_int <= hi:
                return True
    return False


def _resolve_field_alias(name: str, all_keys):
    """Resolve a short field alias (e.g. 'stopBits') to a full config key.

    Falls back to the literal name if it's already a full key, or if no
    unique suffix match exists. Used by @combo: parsing.
    """
    if name in all_keys:
        return name
    matches = [k for k in all_keys if k.endswith("." + name)]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(
        f"Combo field alias '{name}' is ambiguous or unknown "
        f"(matches: {matches}). Use the full config key path."
    )


def _parse_combo_line(line: str, all_keys):
    """Parse an '@combo:' line into (resolved_keys, allowed_combos).

    Format:
        @combo:stopBits,parity=2,0|1,2|1,1
      ->
        keys = [<full key for stopBits>, <full key for parity>]
        allowed_combos = [('2','0'), ('1','2'), ('1','1')]
    """
    body = line[len("@combo:"):]
    if "=" not in body:
        raise ValueError(f"Malformed @combo line (missing '='): {line!r}")
    keys_part, combos_part = body.split("=", 1)
    aliases = [a.strip() for a in keys_part.split(",") if a.strip()]
    keys = [_resolve_field_alias(a, all_keys) for a in aliases]
    combos = []
    for combo_str in combos_part.split("|"):
        values = tuple(v.strip() for v in combo_str.split(","))
        if len(values) != len(keys):
            raise ValueError(
                f"@combo arity mismatch: {len(keys)} keys but combo "
                f"{combo_str!r} has {len(values)} values"
            )
        combos.append(values)
    return keys, combos


def compute_accuracy(gold_path: str, pred_path: str):
    """Compare prediction file against golden keyValues file.

    Returns:
        tuple: (correct, total, pct, results)

        results: list of dicts, one per golden field, in display order
                 (correct first, then non_default, then wrong, then excluded):
            {
              "key":        str,            # config-path key
              "status":     str,            # "correct" | "non_default" | "wrong" | "excluded"
              "default":    str,            # first golden value (factory default)
              "alternates": list[str],      # other accepted values from golden
              "predicted":  str | None,     # model output, or None if missing
              "note":       str,            # extra context (e.g. "wire-equivalent")
            }

    Scoring rules:
      - Predicted matches default            -> status="correct"
      - Predicted matches an alternate       -> status="non_default"  (still scored correct)
      - Predicted matches none               -> status="wrong"
      - Wire-address equivalent              -> status="correct", note="wire-equivalent"
      - Excluded keys (Node/Transaction Name)-> status="excluded"     (not scored)

    Golden files use '|' to list alternate accepted values:
        baudRate=9600|2400|4800|19200
      The first value is the factory default; the rest are also acceptable.

    Coupled-field constraints (e.g. stop-bits/parity combos) use '@combo:'
    lines, which override per-field scoring for the listed fields:
        @combo:stopBits,parity=2,0|1,2|1,1
      Predictions matching any listed combo -> all listed fields counted
      correct (note: 'accepted combo'). Otherwise -> all listed fields wrong
      (note: 'combo (X,Y) not supported').
    """
    # Load golden file ourselves so we can preserve @combo: lines
    # (read_set strips/dedupes but doesn't distinguish constraint lines).
    with open(gold_path, "r", encoding="utf-8-sig") as f:
        gold_lines = [ln.strip() for ln in f if ln.strip()]
    prediction = read_set(pred_path)

    # Build dicts: config-key -> raw value string (may contain '|')
    gold_dict = {}
    combo_lines = []
    for line in gold_lines:
        if line.startswith("@combo:"):
            combo_lines.append(line)
        elif "=" in line:
            k, v = line.split("=", 1)
            gold_dict[k] = v

    pred_dict = {}
    for line in prediction:
        if "=" in line:
            k, v = line.split("=", 1)
            pred_dict[k] = v

    # Parse combo constraints (resolve aliases against the keys present in golden)
    combos = []  # list of (keys: list[str], allowed: list[tuple[str,...]])
    for line in combo_lines:
        keys, allowed = _parse_combo_line(line, list(gold_dict.keys()))
        combos.append((keys, allowed))
    combo_controlled_keys = {k for keys, _ in combos for k in keys}

    # Pre-compute wire-address equivalence
    # Use the golden DEFAULT value (first '|'-segment) for this check.
    wire_equivalent_keys = set()
    gold_addr_mode_default, _ = _parse_golden_values(gold_dict.get(_KEY_ADDR_MODE, ""))
    gold_start_addr_default, _ = _parse_golden_values(gold_dict.get(_KEY_START_ADDR, ""))
    gold_wires = _to_wire_addresses(gold_addr_mode_default, gold_start_addr_default)
    pred_wires = _to_wire_addresses(
        pred_dict.get(_KEY_ADDR_MODE),
        pred_dict.get(_KEY_START_ADDR),
    )
    if gold_wires and pred_wires and gold_wires & pred_wires:
        wire_equivalent_keys.add(_KEY_ADDR_MODE)
        wire_equivalent_keys.add(_KEY_START_ADDR)

    # Build per-field result records
    correct_rows = []
    non_default_rows = []
    wrong_rows = []
    excluded_rows = []

    correct = 0
    total = 0

    # Combo evaluation (overrides per-field scoring for these keys)
    combo_results = {}  # key -> result dict
    for keys, allowed in combos:
        pred_tuple = tuple(pred_dict.get(k, "") for k in keys)
        # Display string for the combo (e.g. "(stopBits=1, parity=2)")
        pred_str = "(" + ", ".join(
            f"{PARAMETER_NAMES.get(k, k.split('.')[-1])}={pred_dict.get(k, 'NOT FOUND')}"
            for k in keys
        ) + ")"
        if pred_tuple in allowed:
            note = f"accepted combo {pred_str}"
            for k in keys:
                if k in EXCLUDED_KEYS:
                    continue
                default, alternates = _parse_golden_values(gold_dict.get(k, ""))
                combo_results[k] = {
                    "key": k, "status": "correct",
                    "default": default, "alternates": alternates,
                    "predicted": pred_dict.get(k), "note": note,
                }
        else:
            allowed_str = " | ".join("(" + ",".join(c) + ")" for c in allowed)
            note = f"combo {pred_str} not supported (allowed: {allowed_str})"
            for k in keys:
                if k in EXCLUDED_KEYS:
                    continue
                default, alternates = _parse_golden_values(gold_dict.get(k, ""))
                combo_results[k] = {
                    "key": k, "status": "wrong",
                    "default": default, "alternates": alternates,
                    "predicted": pred_dict.get(k), "note": note,
                }

    for key in sorted(gold_dict.keys()):
        raw_gold = gold_dict[key]
        default, alternates = _parse_golden_values(raw_gold)
        predicted = pred_dict.get(key)

        if key in EXCLUDED_KEYS:
            excluded_rows.append({
                "key": key,
                "status": "excluded",
                "default": default,
                "alternates": alternates,
                "predicted": predicted,
                "note": "naming convention; not scored",
            })
            continue

        # Combo-controlled key: use combo result, skip per-field scoring
        if key in combo_results:
            row = combo_results[key]
            total += 1
            if row["status"] == "correct":
                correct += 1
                correct_rows.append(row)
            else:
                wrong_rows.append(row)
            continue

        total += 1

        if predicted == default:
            correct += 1
            correct_rows.append({
                "key": key, "status": "correct",
                "default": default, "alternates": alternates,
                "predicted": predicted, "note": "",
            })
        elif predicted is not None and _matches_alternate(predicted, alternates):
            correct += 1
            non_default_rows.append({
                "key": key, "status": "non_default",
                "default": default, "alternates": alternates,
                "predicted": predicted,
                "note": f"accepted alternate; default was {default}",
            })
        elif key in wire_equivalent_keys and predicted is not None:
            correct += 1
            correct_rows.append({
                "key": key, "status": "correct",
                "default": default, "alternates": alternates,
                "predicted": predicted, "note": "wire-equivalent",
            })
        else:
            wrong_rows.append({
                "key": key, "status": "wrong",
                "default": default, "alternates": alternates,
                "predicted": predicted, "note": "",
            })

    results = correct_rows + non_default_rows + wrong_rows + excluded_rows
    pct = 100 * correct / total if total else 0.0
    return correct, total, pct, results


STATUS_LABEL = {
    "correct":     "Correct",
    "non_default": "Non-default match",
    "wrong":       "Wrong",
    "excluded":    "Excluded",
}


def _format_predicted_cell(row):
    """Format the 'Predicted: ...' cell for a result row."""
    pred = row["predicted"]
    if pred is None:
        pred_text = "NOT FOUND"
    else:
        pred_text = pred
    status = row["status"]
    if status == "correct":
        tag = row["note"] if row["note"] else "default"
    elif status == "non_default":
        tag = row["note"]
    elif status == "excluded":
        tag = row["note"]
    elif status == "wrong":
        tag = row["note"]  # may be empty for non-combo wrongs
    else:
        tag = ""
    return f"Predicted: {pred_text} [{tag}]" if tag else f"Predicted: {pred_text}"


# CLI entrypoint (only runs when executed directly)
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=r"golden_keyValues/SDM120_keyValues.txt")
    ap.add_argument("--prediction", default=r"pred.txt")
    ap.add_argument("--out", default="results2.csv")
    ap.add_argument("--manual", default="SDM120-MODBUS_Protocol.pdf")
    ap.add_argument("--model_name", default="gpt-5.1")
    args = ap.parse_args()

    correct, total, pct, results = compute_accuracy(args.gold, args.prediction)

    print(f"Accuracy: {pct:.2f}% ({correct}/{total})  [Node Name + Transaction Name excluded]")
    for row in results:
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
        print(f"  [{label}] {name}: predicted={pred_display}{suffix}")

    out_path = Path(args.out)
    need_header = not Path(args.out).exists()
    with out_path.open("a", newline="", encoding="utf-8") as f:
        write = csv.DictWriter(f, fieldnames=["manual", "ai_model", "accuracy", "accuracy_percent"])
        if need_header:
            write.writeheader()

        write.writerow({
            "manual": args.manual,
            "ai_model": args.model_name,
            "accuracy": f"{correct}/{total}",
            "accuracy_percent": f"{pct:.2f}%"
        })

        for row in results:
            label = STATUS_LABEL[row["status"]]
            exp_line = f"{row['key']}={row['default']}"
            write.writerow({
                "manual": "",
                "ai_model": "",
                "accuracy": f"{label}: {readable_param(exp_line)}",
                "accuracy_percent": _format_predicted_cell(row),
            })

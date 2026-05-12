"""
fc16_echo.py - deterministic FC 16 (Write Multiple Registers) post-processor.

The LLM extraction prompts only cover the FC 3 / FC 4 (Read) template.
When the extracted request function code is 16, the response echo fields and
two extra request fields follow a different rule that we can derive from the
values the LLM already produced. This module rewrites the merged dict in place
and returns the correct ordered key list for output.

FC 16 semantics (Modbus spec):
  Request  : [0]=addr, [1]=FC(16), [2]=startAddr, [3]=qty,
             [4]=byteCount=qty*2, [5]=payload (dataLength=qty*2), [6]=CRC
  Response : [0]=addr, [1]=FC(16), [2]=startAddr echo, [3]=qty echo, [4]=CRC
"""

REQ_FC_KEY   = "subnetwork.nodes[0].transactions[0].frames[0].frameObjects[1].properties.data"
REQ_ADDR_KEY = "subnetwork.nodes[0].transactions[0].frames[0].frameObjects[2].properties.data"
REQ_QTY_KEY  = "subnetwork.nodes[0].transactions[0].frames[0].frameObjects[3].properties.data"
REQ_BCNT_KEY = "subnetwork.nodes[0].transactions[0].frames[0].frameObjects[4].properties.data"
REQ_BLEN_KEY = "subnetwork.nodes[0].transactions[0].frames[0].frameObjects[5].properties.dataLength"

RESP_FC_KEY   = "subnetwork.nodes[0].transactions[0].frames[1].frameObjects[1].properties.data"
RESP_ADDR_KEY = "subnetwork.nodes[0].transactions[0].frames[1].frameObjects[2].properties.data"
RESP_QTY_KEY  = "subnetwork.nodes[0].transactions[0].frames[1].frameObjects[3].properties.data"
RESP_DLEN_KEY = "subnetwork.nodes[0].transactions[0].frames[1].frameObjects[3].properties.dataLength"

# Canonical Read (FC 3/4) 15-key template - shared with the modular-rag scripts.
READ_KEYS = [
    "subnetwork.properties.physicalStandard",
    "subnetwork.properties.baudRate",
    "subnetwork.properties.dataBits",
    "subnetwork.properties.parity",
    "subnetwork.properties.stopBits",
    "subnetwork.nodes[0].properties.nodeAddress",
    "subnetwork.nodes[0].properties.name",
    "subnetwork.nodes[0].properties.modbusAddressingMode",
    "subnetwork.nodes[0].transactions[0].properties.name",
    "subnetwork.nodes[0].transactions[0].frames[0].frameObjects[1].properties.data",
    REQ_ADDR_KEY,
    REQ_QTY_KEY,
    RESP_FC_KEY,
    RESP_ADDR_KEY,
    RESP_DLEN_KEY,
]


def _as_int(value):
    """Parse LLM string value to int, or None if not a number."""
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() == "not found":
        return None
    try:
        return int(s, 0)  # supports "16", "0x10", "0b10000"
    except (ValueError, TypeError):
        return None


def apply_fc16_echo(merged, read_keys):
    """
    If merged describes an FC 16 transaction, mutate merged to add/overwrite
    FC-16-specific fields and return the ordered key list for output.
    Otherwise, return read_keys unchanged.

    merged    : dict[str, str] keyed by dotted-path, values are strings
    read_keys : list[str] - the Read (FC 3/4) template key order

    Returns   : list[str] - ordered output keys
    """
    req_fc = _as_int(merged.get(REQ_FC_KEY))
    if req_fc != 16:
        return list(read_keys)

    qty = _as_int(merged.get(REQ_QTY_KEY))
    req_addr = merged.get(REQ_ADDR_KEY, "not found")

    byte_count = str(qty * 2) if qty is not None else "not found"

    merged[REQ_BCNT_KEY] = byte_count
    merged[REQ_BLEN_KEY] = byte_count
    merged[RESP_FC_KEY]  = "16"
    merged[RESP_ADDR_KEY] = req_addr
    merged[RESP_QTY_KEY]  = str(qty) if qty is not None else "not found"
    merged.pop(RESP_DLEN_KEY, None)

    # Build FC 16 key order: read_keys, minus RESP_DLEN_KEY,
    # with REQ_BCNT/BLEN inserted after REQ_QTY, and RESP_QTY appended at end.
    out = []
    for k in read_keys:
        if k == RESP_DLEN_KEY:
            continue
        out.append(k)
        if k == REQ_QTY_KEY:
            out.append(REQ_BCNT_KEY)
            out.append(REQ_BLEN_KEY)
    out.append(RESP_QTY_KEY)
    return out


def postprocess_fc16_lines(lines, read_keys=None):
    """
    Convenience wrapper for scripts that produce raw "key=value" lines from a
    single LLM call (no merged dict). Parses, applies apply_fc16_echo, emits
    lines in the correct order.

    lines     : iterable of "key=value" strings (other lines are passed through
                untouched at the end).
    read_keys : optional Read (FC 3/4) template key order. Defaults to READ_KEYS.

    Returns   : list[str] of "key=value" lines.
    """
    if read_keys is None:
        read_keys = READ_KEYS

    merged = {}
    passthrough = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if "=" not in s or not s.startswith("subnetwork"):
            passthrough.append(s)
            continue
        key, _, value = s.partition("=")
        merged[key.strip()] = value.strip()

    output_keys = apply_fc16_echo(merged, read_keys)
    out = [f"{k}={merged.get(k, 'not found')}" for k in output_keys]
    return out + passthrough

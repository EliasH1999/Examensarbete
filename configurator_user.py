"""Tier-0 baseline: prompt mirrors the Anybus Communicator configurator UI.

The LLM is asked the EXACT field labels a real user sees in the online
configurator (Physical stand, Baud rate, Data bits, Parity, Stop bits,
Node address, Name, Address format, Transaction name, Modbus transaction,
Address, Quantity). Dropdown choices are listed verbatim - that's not a
hint, it's what the GUI shows.

A deterministic Python normalizer maps the LLM's plain-text answers to
the integer/path schema that accuracy.py expects. All Modbus protocol
knowledge (enum mapping, byte-count = quantity*2, response FC = request FC)
lives in the normalizer, NOT in the prompt.

This isolates "raw LLM extraction ability" - the prompt knows nothing
about the Modbus protocol beyond what the configurator UI shows.
"""

import argparse
import re

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader


# CLI
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--manual", type=str, default=None)
_parser.add_argument("--question", type=str, default=None)
_parser.add_argument("--output", type=str, default="pred.txt")
_cli_args, _ = _parser.parse_known_args()


# Load PDF
_manual_path = _cli_args.manual or "random_manuals/SDM120-MODBUS_Protocol.pdf"
docs = PyMuPDFLoader(_manual_path).load()
manual_text = ""
for i, doc in enumerate(docs):
    if doc.page_content:
        manual_text += f"Page {i+1}: {doc.page_content}\n"


# Model + UI-mirroring prompt
model = init_chat_model("gpt-5.1", temperature=0)

SYSTEM = """You are helping a user fill in the Anybus Communicator online configurator
for a Modbus RTU device. Read the device manual and answer each field below
using the manual. If a value is not stated, write "not found".

Output one line per field, exactly in this format <field>=<value>:

Communication / Basic settings:
  Physical stand        (RS-232 or RS-485)
  Baud rate             (e.g. 9600)
  Data bits             (7 or 8)
  Parity                (None, Odd, or Even)
  Stop bits             (1 stop bit or 2 stop bits)

Node settings:
  Node address          (1-247)
  Name                  (the device model/product name)
  Address format        (Address (0,1,2,...), Register (1,2,3,...),
                         Modicon (00001/10001/30001/40001),
                         or Modicon extended (000001/100001/300001/400001))

Transaction settings:
  Transaction name      (the name of the register being read)
  Modbus transaction    (Read Coils (1), Read Discrete Inputs (2),
                         Read Holding Registers (3), Read Input Registers (4),
                         Write Single Coil (5), Write Single Register (6),
                         Write Multiple Coils (15), Write Multiple Registers (16),
                         Mask Write Register (22), Read Write Multiple Registers (23))
  Address               (the register address from the manual)
  Quantity              (number of 16-bit registers, 1-125)

Output ONLY these 12 lines, exactly <field>=<value>, no extra text.
"""

prompt = ChatPromptTemplate.from_messages(
    [("system", SYSTEM),
     ("user", "MANUAL: {manual_text} \n\n QUESTION: {question}")]
)
chain = prompt | model

# NORMALIZER - pure Python, no LLM, deterministic.
# Maps the configurator-style answers to the schema accuracy.py expects.
NF = "not found"


def _first_int(s: str):
    m = re.search(r"-?\d+", s)
    return int(m.group()) if m else None


def _norm_physical(v: str):
    s = v.lower()
    if "485" in s:
        return 1
    if "232" in s:
        return 0
    return None


def _norm_baud(v: str):
    return _first_int(v)


def _norm_databits(v: str):
    n = _first_int(v)
    if n == 7:
        return 0
    if n == 8:
        return 1
    return None


def _norm_parity(v: str):
    s = v.strip().lower()
    if s.startswith("n"):
        return 0
    if s.startswith("o"):
        return 1
    if s.startswith("e"):
        return 2
    return None


def _norm_stopbits(v: str):
    n = _first_int(v)
    if n == 1:
        return 0
    if n == 2:
        return 1
    return None


def _norm_addr_format(v: str):
    s = v.lower()
    if "extended" in s or "6-digit" in s:
        return 3
    if "modicon" in s or "5-digit" in s:
        return 2
    if "register" in s or "1-based" in s:
        return 1
    if "address" in s or "0-based" in s:
        return 0
    return None


def _norm_transaction_fc(v: str):
    """Modbus transaction dropdown -> function code integer.
    The configurator displays the FC number in parens, e.g. 'Read Holding Registers (3)'.
    Prefer parsing that number; fall back to name matching.
    """
    m = re.search(r"\((\d+)\)", v)
    if m:
        return int(m.group(1))
    s = v.lower()
    table = [
        ("read coils", 1),
        ("read discrete", 2),
        ("read holding", 3),
        ("read input", 4),
        ("write single coil", 5),
        ("write single register", 6),
        ("write multiple coils", 15),
        ("write multiple registers", 16),
        ("mask write", 22),
        ("read write multiple", 23),
    ]
    for needle, fc in table:
        if needle in s:
            return fc
    return _first_int(v)


# Mapping: configurator label (lowercased, no spaces) -> normalizer
FIELD_NORMALIZERS = {
    "physicalstand": _norm_physical,
    "physicalstandard": _norm_physical,
    "baudrate": _norm_baud,
    "databits": _norm_databits,
    "parity": _norm_parity,
    "stopbits": _norm_stopbits,
    "nodeaddress": _first_int,
    "name": lambda v: v.strip(),
    "addressformat": _norm_addr_format,
    "transactionname": lambda v: v.strip(),
    "modbustransaction": _norm_transaction_fc,
    "address": _first_int,
    "quantity": _first_int,
}


def _parse_llm_lines(text: str) -> dict:
    """Parse '<field>=<value>' lines, normalize values."""
    out = {}
    for raw in text.splitlines():
        if "=" not in raw:
            continue
        k, v = raw.split("=", 1)
        key = re.sub(r"[\s_\-()]", "", k).lower()
        val = v.strip()
        if key not in FIELD_NORMALIZERS:
            continue
        if val.lower() in {"", "not found", "n/a", "none stated", "unknown"} and key != "parity":
            # leave as not found; skip parity because "None" is a real value there
            out[key] = NF
            continue
        normalized = FIELD_NORMALIZERS[key](val)
        out[key] = NF if normalized is None else normalized
    return out


def _build_schema_lines(parsed: dict) -> list:
    """Map parsed/normalized fields to the 15 schema lines accuracy.py scores."""
    g = lambda k: parsed.get(k, NF)

    fc = g("modbustransaction")
    qty = g("quantity")
    byte_count = qty * 2 if isinstance(qty, int) else NF

    pairs = [
        ("subnetwork.properties.physicalStandard", g("physicalstand") if "physicalstand" in parsed else g("physicalstandard")),
        ("subnetwork.properties.baudRate",         g("baudrate")),
        ("subnetwork.properties.dataBits",         g("databits")),
        ("subnetwork.properties.parity",           g("parity")),
        ("subnetwork.properties.stopBits",         g("stopbits")),
        ("subnetwork.nodes[0].properties.nodeAddress",         g("nodeaddress")),
        ("subnetwork.nodes[0].properties.name",                g("name")),
        ("subnetwork.nodes[0].properties.modbusAddressingMode", g("addressformat")),
        ("subnetwork.nodes[0].transactions[0].properties.name", g("transactionname")),
        ("subnetwork.nodes[0].transactions[0].frames[0].frameObjects[1].properties.data", fc),
        ("subnetwork.nodes[0].transactions[0].frames[0].frameObjects[2].properties.data", g("address")),
        ("subnetwork.nodes[0].transactions[0].frames[0].frameObjects[3].properties.data", qty),
        ("subnetwork.nodes[0].transactions[0].frames[1].frameObjects[1].properties.data", fc),
        ("subnetwork.nodes[0].transactions[0].frames[1].frameObjects[2].properties.data", byte_count),
        ("subnetwork.nodes[0].transactions[0].frames[1].frameObjects[3].properties.dataLength", byte_count),
    ]
    return [f"{k}={v}" for k, v in pairs]


# Run
def _run(question: str, output_path: str):
    response = chain.invoke({"manual_text": manual_text, "question": question})
    from usage_logger import log_usage; log_usage(response, pass_name="main")
    parsed = _parse_llm_lines(response.content)
    lines = _build_schema_lines(parsed)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if _cli_args.manual and _cli_args.question:
    _run(_cli_args.question, _cli_args.output)
else:
    while True:
        q = input("Ask a question about the PDF: ")
        if q.lower() == "exit":
            break
        _run(q, _cli_args.output)

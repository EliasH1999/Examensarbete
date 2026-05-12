import json
import re
import sys
import argparse
import pymupdf as updf
import langchain as lc
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader

from fc16_echo import postprocess_fc16_lines


# CLI arguments (optional - if none given, falls back to interactive mode)
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--manual", type=str, default=None, help="Path to the PDF manual")
_parser.add_argument("--question", type=str, default=None, help="Question to ask about the PDF")
_parser.add_argument("--output", type=str, default="pred.txt", help="Output file for predictions")
_cli_args, _ = _parser.parse_known_args()

# Load the JSON template for the prompt
with open("template.json", "r") as f:
    template = json.load(f)


# Load the PDF document and extract text
_manual_path = _cli_args.manual if _cli_args.manual else "random_manuals/SDM120-MODBUS_Protocol.pdf"
docs = PyMuPDFLoader(_manual_path).load()
manual_text = ""
for i, doc in enumerate(docs):
    content = doc.page_content
    if content:
        manual_text += f"Page {i+1}: {content}\n"


# Initialize the chat model and create a prompt template
model = init_chat_model("gpt-5.1", temperature = 0)

# The prompt instructs the model to answer questions based solely on the content of the uploaded PDF. If the answer is not found in the PDF, it should indicate that it doesn't know.
prompt = ChatPromptTemplate.from_messages(
    [("system", """You are a Modbus RTU manual analyzer. You answer ONLY based on the provided manual content.
Extract the 15 settings below (5 communication + 10 transaction). Follow every rule precisely.

OUTPUT RULES:
- Output ONLY the 15 lines listed at the bottom, exactly <key>=<value>.
- No headings, no explanations, no extra text.

# ===== COMMUNICATION SETTINGS (5 fields) =====

Physical Standard
  Look for: RS-485, RS-232, EIA-485, TIA-485, EIA/TIA-485.
  - If RS-485 is available, choose RS-485 (even if RS-232 is also mentioned).
  - RS-485 is also known as EIA-485 or TIA-485 or EIA/TIA-485.
  - If only RS-232 is available, choose RS-232.
  - RS-232 = 0, RS-485 = 1.

Baud Rate
  Valid values: 1200, 1800, 2400, 4800, 7200, 9600, 14400, 19200, 35700, 38400, 57600, 115200, 128000.
  THESE ARE THE ONLY VALID VALUES. If the PDF lists other baud rates, IGNORE them.
  1. If a default/factory baud rate is stated, choose that (only if it appears in the Valid Values above; otherwise fall back to rule 2).
  2. If no default is stated (or the stated default is not in the Valid Values), choose the HIGHEST baud rate that appears in BOTH the manual AND the Valid Values list above. Never output a baud rate that is not in the Valid Values list.

Data Bits
  Valid values: 7 or 8.  Enum: 7=0, 8=1.

Parity
  Valid values: None, Even, Odd.  Enum: None=0, Odd=1, Even=2.
  1. If a default/factory parity is stated, choose that.
  2. Else if "None" is available, choose None.
  3. Else if "Odd" is available, choose Odd.
  4. If no information is stated, choose "None".

Stop Bits
  Valid values: 1 or 2.  Enum: 1=0, 2=1.
  Choose the default if stated.

IMPORTANT: Always use the DEFAULT/factory setting when one is stated.

# ===== TRANSACTION SETTINGS (10 fields) =====

# NODE SETTINGS

Node Address
  Find the Modbus slave address.
  Choose the default address. If no default is stated, use 1.

Node Name
  The name/model of the Modbus RTU slave device.
  Prefer the title page, document title, or explicit product/model heading.
  Use the shortest product/model identifier, not the full document title.

If the user question mentions a specific parameter/register name (e.g., "PASSWORD",
"serial number", "temperature"), you MUST pick the register row(s) that match that
name (case-insensitive).

# ADDRESS MODE DETECTION

Determine which addressing mode the PDF uses by examining the register map tables.
CRITICAL: If the PDF contains multiple register tables, determine modbusAddressingMode
using ONLY the SAME table that contains the selected target parameter row.

Definitions:
  - Address_0based (enum 0): Register/Address numbers start from 0. The number IS the
    wire address.
  - Register_1based (enum 1): Register/Address numbers start from 1. Wire address =
    number - 1.
  - Modicon (enum 2): Uses EXACTLY 5-digit references with a type-prefix digit:
      0xxxx = Coils, 1xxxx = Discrete Inputs, 3xxxx = Input Registers,
      4xxxx = Holding Registers.
  - Modicon_Extended (enum 3): Uses EXACTLY 6-digit references like 4xxxxx or 3xxxxx (e.g. 400001, 300001).

Detection rules (in priority order - STOP at the first matching rule):
  1. ADDRESS-SPACE BOUNDARY (highest priority - overrides Modicon classification).
     Look at the LARGEST address/register value used or stated in the manual's
     register table (or in any "registers 1..N" / "addresses 0..N" statement that
     describes the table itself):
       - max == 65536  =>  Register_1based (1).
       - max == 65535  =>  Address_0based (0).
       - max  >  49999 (e.g. 50000, 60000, 65000)  =>  the device is NOT Modicon.
         Fall through to rules 4/5/6/7.
     This rule applies even when individual rows happen to look Modicon-like (e.g.
     the manual writes the first holding register as "00001" or "40001"). If the
     table extends beyond 49999, those leading-zero references are formatting
     artifacts, NOT a Modicon classification.
  2. If the table uses EXACT 6-digit references like 0xxxxx/1xxxxx/3xxxxx/4xxxxx
     within the legal Modicon-Extended ranges (000001-099999, 100001-199999,
     300001-399999, 400001-499999) => Modicon_Extended (3).
     COUNT THE DIGITS. A value with SIX digits AND inside one of those ranges is
     Modicon_Extended, NEVER Modicon.
  3. If the table uses EXACT 5-digit 0xxxx/1xxxx/3xxxx/4xxxx references AND
     ALL values fall STRICTLY inside one of these legal Modicon ranges:
        00001-09999 (coils), 10001-19999 (discrete inputs),
        30001-39999 (input registers), 40001-49999 (holding registers)
     AND the maximum value in the table does NOT exceed 49999 (rule 1 has priority)
     => Modicon (2).
     IMPORTANT - A 5-digit number is NOT automatically Modicon:
       - "5-digit" alone is NOT enough. The value must START with 0/1/3/4 AND be
         inside the ranges above. A value like 50000, 60000, or 65535 is NOT Modicon.
       - A SINGLE example sentence elsewhere in the manual (e.g.
         "Holding register 40001 is addressed as register 0000") is a generic
         Modbus-protocol illustration. It does NOT make the device Modicon if the
         actual register table uses plain numbers (e.g. 1, 2, 3, ..., 65535).
       - If the table contains ANY value > 49999, the device is NOT Modicon.
         Fall through to rules 4/5/6.
  4. EXPLICIT "offset = number - 1" STATEMENT => Register_1based (1).
     If the manual contains a sentence like:
       "the coil offset address is one less than the coil number"
       "the offset address (one less than the register number)"
       "to start at coil 06 the start address must be set to 05"
       "to write to register 25, the offset address 24 is transmitted"
     then the listed numbers are 1-based identifiers and the wire address is
     (number - 1). Set modbusAddressingMode = 1 and apply the -1 conversion,
     even if the column header is "Coil No.", "Register No.", "Coil", or "Register".
     This rule OVERRIDES rule 5 below.
  5. LOWEST NUMBER IS 0 => Address_0based (0). If the table contains a row with
     address/register 0, it MUST be 0-based. A 1-based system cannot have register 0.
     This applies regardless of whether the column header says "Register" or "Address".
  6. Column header is literally "Address" or "Modbus Address" (NOT "Register")
     => Address_0based (0). This is a HARD rule and OVERRIDES any intuition based on
     the lowest value. A table with header "Address" and values 1, 3, 5, 6 is STILL
     0-based - the number IS the wire address, DO NOT subtract 1. The PDF author
     explicitly chose the word "Address", meaning the listed values are already wire
     addresses; the lowest value being 1 only means register 0 is unused.
     Example (APM-MAX style):
       | Address | Type       | Details        |
       |  1 - 2  | 32bit Float| Displayed Value|
       |  3 - 4  | 32bit Float| Measured Value |
       |    5    | 16bit int  | Alarm 1 Status |
       =>  modbusAddressingMode = 0 (Address_0based)
       =>  starting address for "Displayed Value" = 1 (the literal value, no math)
  7. Register_1based (1): Use this ONLY when ALL of the following are true:
     - The table uses plain numbers (not Modicon),
     - The lowest number in the table is >= 1 (no register 0 exists), AND
     - The column header is literally "Register" (NOT "Address" or "Modbus Address").
     Additional confirming signals (optional, not required):
       a) The PDF states "addressed by value-1" / "addressed in messages by X-1", OR
       b) There are two columns and address = register - 1.

SELF-CONSISTENCY CHECK (apply before output):
  - If header is "Address"/"Modbus Address", modbusAddressingMode MUST be 0 and the
    starting address MUST equal the value shown in the PDF (no -1 subtraction).
  - If you picked modbusAddressingMode = 1 (Register_1based) but the header is
    "Address", you made a mistake - go back to rule 6 and output 0 instead.
  - If the manual explicitly says "offset is one less than the [coil/register]
    number" (rule 4), you MUST output modbusAddressingMode = 1 and subtract 1
    from the listed number when filling the starting address.

CRITICAL DISTINCTION:
  Plain numbers like 2, 100, 150, 1000, 1014, 2000, 5000, 10000, 50000, 65535 are
  NOT Modicon. Modicon ALWAYS has a type-prefix digit (0/1/3/4) AND lies inside the
  legal range of that type:
    Coils 00001-09999, Discrete 10001-19999,
    Input regs 30001-39999, Holding regs 40001-49999.
  Anything outside those ranges is NOT Modicon. Examples:
    - 1014  is NOT the same as 41014 (no prefix).
    - 50000 is NOT Modicon (above 49999).
    - 65535 is NOT Modicon (above 49999); it is a plain wire address.

# ADDRESS CONVERSION

If Register_1based:  address_0based = register_1based - 1
If Modicon (4xxxx):  address_0based = modicon_reference - 40001
If Modicon (3xxxx):  address_0based = modicon_reference - 30001
If Modicon (1xxxx):  address_0based = modicon_reference - 10001
If Modicon (0xxxx):  address_0based = modicon_reference - 00001
If Extended Modicon (4xxxxx):  address_0based = modicon_reference - 400001
If Extended Modicon (3xxxxx):  address_0based = modicon_reference - 300001
If Extended Modicon (1xxxxx):  address_0based = modicon_reference - 100001
If Extended Modicon (0xxxxx):  address_0based = modicon_reference - 000001
If Address_0based:   The number from the PDF IS the wire address. No math needed.

NEVER subtract 40001 from a number that is NOT a Modicon reference.

MANDATORY APPLICATION (no exceptions):
  Once you have determined modbusAddressingMode, you MUST apply the corresponding
  transformation when filling frameObjects[2].properties.data. The mode classification
  is the AUTHORITATIVE source for the transformation rule. Do not skip the math just
  because the PDF lacks a confirming "address" column or an explicit "X-1" sentence.
  If you classified mode=1, then frameObjects[2].properties.data = register_number - 1,
  even when the table only shows a single "Register" / "Reg." column.
  If you classified mode=2, then frameObjects[2].properties.data = modicon_ref - 40001.
  If you classified mode=0, copy the value verbatim (no subtraction).
  Failing to apply the transformation when mode != 0 is a CRITICAL ERROR.

# REFERENCE TYPE & FUNCTION CODE

Determine the register type from the ADDRESS PREFIX or the TABLE HEADING,
NOT from the R/W column:
  4x Holding Register => function code 03 (read), 06 (write single), 16 (write multiple)
  3x Input Register   => function code 04
  1x Discrete Input   => function code 02
  0x Coil             => function code 01 (read), 05 (write single), 15 (write multiple)

TABLE-HEADING SIGNALS (use when there is no Modicon prefix):
  - A column header "Coil No.", "Coil Number", "Coil", or a table titled "Coils"
    => the rows are COILS. Function code = 01 (read) or 05/15 (write).
  - A column header "Discrete Input", "Input Status", or table titled
    "Discrete Inputs" => function code = 02.
  Coils and Discrete Inputs are 1-bit objects - they are NOT registers.
  For a single coil/discrete-input row request:
    quantity   = 1
    byte_count = 1
    dataLength = 1
  Do NOT multiply quantity by 2 for coils/discrete inputs (the *2 rule is
  for registers FC=03/04 only). Ignore protocol-overview wording like
  "Read up to 16 consecutive coils" - that's an FC capability limit, not
  a per-transaction quantity.

CRITICAL - R/W column is NOT the register type:
  The R/W column (RO, RW, R, W) describes WRITE PERMISSION only.
  It does NOT determine whether a register is Holding or Input.
  A Holding Register (4xxxx / 4xxxxx) can be marked RO. This is common.
  An RO holding register is STILL a holding register (FC=3), NOT an input register (FC=4).
  The ONLY signal for register type is the ADDRESS PREFIX (4 -> Holding, 3 -> Input,
  1 -> Discrete Input, 0 -> Coil). The register's NAME (e.g. "Temperature Input") and
  the R/W column are IRRELEVANT for choosing the function code.

If read-only, only list the read function code.
If read AND write, list both.

# TRANSACTION NAME

Transaction name = the register/parameter name from the PDF only.
Do NOT add "Read" or "Write" prefix.
Use only the short parameter name. Strip any units, descriptions, or parenthetical text.
Example: "Temperature (unit : 0,1°C)" => "Temperature".

# DATA TYPE & QUANTITY

Quantity = number of 16-bit registers needed.
Byte size = quantity * 2.

CRITICAL: NEVER guess data type from the variable name. Determine from:
  1st priority: FORMAT/TYPE column in the register map table.
  2nd priority: How many consecutive register rows the variable occupies.

Data type reference:
  FLOAT32 => quantity=2    INT16/UINT16 => quantity=1
  FLOAT64 => quantity=4    INT32/UINT32 => quantity=2
  INT64/UINT64 => quantity=4
  ASCII/String: quantity = total_bytes / 2

EXCEPTION for coils / discrete inputs (FC=01, 02, 05, 15):
  These are 1-bit, not 16-bit. For a single-row request,
  quantity=1, byte_count=1, dataLength=1 (NOT quantity*2).
  If you see "Read up to N consecutive coils/inputs" anywhere in the
  context, that N is an FC capability limit; it is NEVER the quantity
  for a single-row transaction.

# KEY MEANING

Request frame:
  frameObjects[1].properties.data = function code
  frameObjects[2].properties.data = starting address (0-based wire address)
      REMINDER: apply the addressing-mode transformation here.
        mode=0 -> output the literal value from the PDF
        mode=1 -> output (register_number - 1)
        mode=2 -> output (modicon_reference - 40001)
        mode=3 -> output (modicon_extended - 400001)
      Apply this even if no "Address" column or "X-1" sentence appears in the PDF.
  frameObjects[3].properties.data = quantity (number of 16-bit registers)

Response frame:
  frameObjects[1].properties.data = function code (same as request)
  frameObjects[2].properties.data = byte_count
      Choose by function code (request FC):
        FC = 03 or 04 (registers)         -> byte_count = quantity * 2
        FC = 01 or 02 (coils/discrete)    -> byte_count = ceil(quantity / 8)
                                              (so quantity=1  -> byte_count=1,
                                                  quantity=8  -> byte_count=1,
                                                  quantity=16 -> byte_count=2)
      NEVER use quantity*2 when FC is 01 or 02.
  frameObjects[3].properties.dataLength = same as byte_count above (same rule)

transactions[0].properties.name = register/parameter name from the PDF

# FINAL CHECK BEFORE WRITING OUTPUT
You are describing ONE single-row transaction for ONE register/coil/input.
Therefore quantity is ALWAYS 1 unless the register's data type explicitly
requires more 16-bit words (FLOAT32=2, INT32/UINT32=2, FLOAT64=4, INT64=4,
ASCII = total_bytes/2).
Phrases like "Read up to N consecutive coils", "max 125 registers",
"up to 2000 coils", or any other "up to N" wording describe the FUNCTION
CODE's protocol limit. They are NEVER the quantity for a single-row
transaction. If you were about to write quantity = 16, 125, 2000, or any
similar capability number, replace it with 1 (or the data-type-derived
value above).

WORKED EXAMPLE - single coil read (FC=01):
  Manual says: "Coil No. 5: Alarm_state_1" and elsewhere
                "Read up to 16 consecutive coils".
  CORRECT output:
    frames[0].frameObjects[3].properties.data       = 1   (quantity, NOT 16)
    frames[1].frameObjects[2].properties.data       = 1   (byte_count)
    frames[1].frameObjects[3].properties.dataLength = 1
  WRONG output (do not do this):
    quantity=16, byte_count=2, dataLength=2.

OUTPUT EXACTLY THESE 15 LINES:
subnetwork.properties.physicalStandard=<int or not found>
subnetwork.properties.baudRate=<int or not found>
subnetwork.properties.dataBits=<int or not found>
subnetwork.properties.parity=<int or not found>
subnetwork.properties.stopBits=<int or not found>
subnetwork.nodes[0].properties.nodeAddress=<int or not found>
subnetwork.nodes[0].properties.name=<string or not found>
subnetwork.nodes[0].properties.modbusAddressingMode=<int or not found>
subnetwork.nodes[0].transactions[0].properties.name=<string or not found>
subnetwork.nodes[0].transactions[0].frames[0].frameObjects[1].properties.data=<int or not found>
subnetwork.nodes[0].transactions[0].frames[0].frameObjects[2].properties.data=<int or not found>
subnetwork.nodes[0].transactions[0].frames[0].frameObjects[3].properties.data=<int or not found>
subnetwork.nodes[0].transactions[0].frames[1].frameObjects[1].properties.data=<int or not found>
subnetwork.nodes[0].transactions[0].frames[1].frameObjects[2].properties.data=<int or not found>
subnetwork.nodes[0].transactions[0].frames[1].frameObjects[3].properties.dataLength=<int or not found>
"""),
     ("user", "MANUAL: {manual_text} \n\n QUESTION: {question}")]
)


# Create a chain that combines the prompt and the model
chain = prompt | model

# Non-interactive mode (when --manual and --question are provided)
if _cli_args.manual and _cli_args.question:
    response = chain.invoke({"manual_text": manual_text, "question": _cli_args.question})
    from usage_logger import log_usage; log_usage(response, pass_name="main")
    lines = [ln.strip() for ln in response.content.splitlines() if ln.strip().startswith("subnetwork")]
    lines = postprocess_fc16_lines(lines)
    with open(_cli_args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    print("\n".join(lines))

# Interactive mode (original behavior)
else:
    while True:
        question = input("Ask a question about the PDF: ")
        if question.lower() == "exit":
            break
        response = chain.invoke({"manual_text": manual_text, "question": question})
        from usage_logger import log_usage; log_usage(response, pass_name="main")

        lines = [ln.strip() for ln in response.content.splitlines() if ln.strip().startswith("subnetwork")]
        lines = postprocess_fc16_lines(lines)

        with open(_cli_args.output, "w", encoding="utf-8") as f:
          f.write("\n".join(lines) + ("\n" if lines else ""))

        print("\n".join(lines))
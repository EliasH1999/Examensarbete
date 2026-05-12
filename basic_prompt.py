"""Basic-prompt baseline (no Modbus protocol explanation, no enum mapping, no rules).

This is the minimum-engineering baseline used to isolate how much the engineered
prompt and the RAG pipelines actually contribute. The model receives:
  - The full PDF text.
  - A short user question.
  - A bare instruction listing the keys to fill, with NO domain knowledge.

Mirrors the CLI surface of prompt.py so run_benchmark.py can drive it the same way.
"""

import argparse

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader


# CLI arguments
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--manual", type=str, default=None, help="Path to the PDF manual")
_parser.add_argument("--question", type=str, default=None, help="Question to ask about the PDF")
_parser.add_argument("--output", type=str, default="pred.txt", help="Output file for predictions")
_cli_args, _ = _parser.parse_known_args()


# Load PDF
_manual_path = _cli_args.manual if _cli_args.manual else "random_manuals/SDM120-MODBUS_Protocol.pdf"
docs = PyMuPDFLoader(_manual_path).load()
manual_text = ""
for i, doc in enumerate(docs):
    content = doc.page_content
    if content:
        manual_text += f"Page {i+1}: {content}\n"


# Model + minimal prompt
model = init_chat_model("gpt-5.1", temperature=0)

SYSTEM = """You are an extraction assistant. Read the manual and return the requested values.
Output ONLY the lines below in the format <key>=<value>, one per line.
If a value is not stated in the manual, output <key>=not found.
No explanations, no headings, no extra lines.

Use these integer encodings for the enum fields (output the integer, not the word):
  physicalStandard:       RS-232=0, RS-485=1
  parity:                 None=0, Odd=1, Even=2
  stopBits:               1 stop bit=0, 2 stop bits=1
  dataBits:               7 data bits=0, 8 data bits=1
  modbusAddressingMode:   Address_0based=0, Register_1based=1, Modicon=2, Modicon_Extended=3
  function code:          Read Coils=1, Read Discrete Inputs=2, Read Holding Registers=3,
                          Read Input Registers=4, Write Single Coil=5, Write Single Register=6,
                          Write Multiple Coils=15, Write Multiple Registers=16
All other fields are plain integers or strings as they appear in the manual.

What each key means (the keys are template field names, NOT labels from the manual):
  subnetwork.properties.physicalStandard            = the serial physical layer (RS-232 or RS-485)
  subnetwork.properties.baudRate                    = baud rate in bits per second (e.g. 9600, 19200)
  subnetwork.properties.dataBits                    = number of data bits per character (7 or 8)
  subnetwork.properties.parity                      = parity setting (None, Odd, Even)
  subnetwork.properties.stopBits                    = number of stop bits (1 or 2)

  subnetwork.nodes[0].properties.nodeAddress        = Modbus slave address of the device
  subnetwork.nodes[0].properties.name               = the model/product name of the device
  subnetwork.nodes[0].properties.modbusAddressingMode = how the manual writes register numbers
                                                       (see modbusAddressingMode encoding above)

  subnetwork.nodes[0].transactions[0].properties.name           = name of the register being read
  Request frame (master -> slave):
    frames[0].frameObjects[1].properties.data       = function code of the request
    frames[0].frameObjects[2].properties.data       = starting register/wire address (integer)
    frames[0].frameObjects[3].properties.data       = quantity = number of 16-bit registers to read
  Response frame (slave -> master):
    frames[1].frameObjects[1].properties.data       = function code of the response (same as request)
    frames[1].frameObjects[2].properties.data       = byte count = quantity * 2
    frames[1].frameObjects[3].properties.dataLength = data length in bytes = quantity * 2

Sizing reference (number of 16-bit registers per common data type):
  INT16 / UINT16  -> 1   (2 bytes)
  INT32 / UINT32  -> 2   (4 bytes)
  FLOAT32 / IEEE754 single  -> 2   (4 bytes)
  INT64 / UINT64 / FLOAT64  -> 4   (8 bytes)
  ASCII string    -> ceil(characters / 2)

subnetwork.properties.physicalStandard=
subnetwork.properties.baudRate=
subnetwork.properties.dataBits=
subnetwork.properties.parity=
subnetwork.properties.stopBits=
subnetwork.nodes[0].properties.nodeAddress=
subnetwork.nodes[0].properties.name=
subnetwork.nodes[0].properties.modbusAddressingMode=
subnetwork.nodes[0].transactions[0].properties.name=
subnetwork.nodes[0].transactions[0].frames[0].frameObjects[1].properties.data=
subnetwork.nodes[0].transactions[0].frames[0].frameObjects[2].properties.data=
subnetwork.nodes[0].transactions[0].frames[0].frameObjects[3].properties.data=
subnetwork.nodes[0].transactions[0].frames[1].frameObjects[1].properties.data=
subnetwork.nodes[0].transactions[0].frames[1].frameObjects[2].properties.data=
subnetwork.nodes[0].transactions[0].frames[1].frameObjects[3].properties.dataLength=
"""

prompt = ChatPromptTemplate.from_messages(
    [("system", SYSTEM),
     ("user", "MANUAL: {manual_text} \n\n QUESTION: {question}")]
)

chain = prompt | model


def _run(question: str, output_path: str):
    response = chain.invoke({"manual_text": manual_text, "question": question})
    lines = [ln.strip() for ln in response.content.splitlines() if ln.strip().startswith("subnetwork")]
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    print("\n".join(lines))


# Non-interactive mode
if _cli_args.manual and _cli_args.question:
    _run(_cli_args.question, _cli_args.output)

# Interactive mode
else:
    while True:
        question = input("Ask a question about the PDF: ")
        if question.lower() == "exit":
            break
        _run(question, _cli_args.output)

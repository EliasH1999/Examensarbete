import json
import sys

def load(path: str):
    # Try UTF-8 with BOM first, then UTF-16
    for encoding in ["utf-8-sig", "utf-16"]:
        try:
            with open(path, "r", encoding=encoding) as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    # If both fail, raise an error
    raise ValueError(f"Could not decode {path} with any supported encoding")
# Goes through the list with keys and indices to get the value.
def get(obj, path: str):
    for p in path:
        obj = obj[p]    
    return obj  

FIELDS = {
    # Serial
    "baudRate": ["subnetwork", "properties", "baudRate"],
    "parity": ["subnetwork", "properties", "parity"],
    "stopBits": ["subnetwork", "properties", "stopBits"],
    "dataBits": ["subnetwork", "properties", "dataBits"],

    # Node
    "nodeAddress": ["subnetwork", "nodes", 0, "properties", "nodeAddress"],
    "name": ["subnetwork", "nodes", 0, "properties", "name"],
    "modbusAddressingMode": ["subnetwork", "nodes", 0, "properties", "modbusAddressingMode"],

    # Transaction
    "transactionName": ["subnetwork", "nodes", 0, "transactions", 0, "properties", "name"],

    # Request frame
    "requestFunctionCode": ["subnetwork", "nodes", 0, "transactions", 0, "frames", 0, "frameObjects", 1, "properties", "data"],
    "requestStartingAddress": ["subnetwork", "nodes", 0, "transactions", 0, "frames", 0, "frameObjects", 2, "properties", "data"],
    "requestQuantity": ["subnetwork", "nodes", 0, "transactions", 0, "frames", 0, "frameObjects", 3, "properties", "data"],

    # Response frame
    "responseFunctionCode": ["subnetwork", "nodes", 0, "transactions", 0, "frames", 1, "frameObjects", 1, "properties", "data"],
    "responseByteCount": ["subnetwork", "nodes", 0, "transactions", 0, "frames", 1, "frameObjects", 2, "properties", "data"],
    "responseDataLength": ["subnetwork", "nodes", 0, "transactions", 0, "frames", 1, "frameObjects", 3, "properties", "dataLength"],

}
# Loads the output and golden JSON files, and compares the values at the specified paths, printing any mismatches. Usage: python compare.py output.json golden.json
output = load(sys.argv[1])
golden = load(sys.argv[2])

# For each field, get the value from both the output and golden JSON using the specified path, and compare them. If they don't match, print a message indicating the mismatch.
for label, path in FIELDS.items():
    a = get(output, path)
    b = get(golden, path)
    if a != b:
        print(f"Mismatch in {label}: got {a}, expected {b}")


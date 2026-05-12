import json

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)
    
template = load_json("template.json")
changes = load_json("changes.json")

# Serial changes
sp = template["subnetwork"]["properties"]
sp["baudRate"] = changes["serial"]["baudRate"]
sp["parity"] = changes["serial"]["parity_enum"]
sp["stopBits"] = changes["serial"]["stopBits_enum"]
sp["dataBits"] = changes["serial"]["dataBits_enum"]

# Node Changes 
node = template["subnetwork"]["nodes"][0]["properties"]
node["nodeAddress"] = changes["node"]["nodeAddress"]
node["name"] = changes["node"]["name"]
node["modbusAddressingMode"] = changes["serial"]["modbusAddressingMode_enum"]

# Transaction changes
transaction = template["subnetwork"]["nodes"][0]["transactions"][0]
transaction["properties"]["name"] = changes["read"]["name"]

function_code = changes["read"]["function_code"]
address = changes["read"]["starting_address"]
quantity = changes["read"]["quantity"]
byte_count= quantity * 2


# Request frame changes
transaction["frames"][0]["frameObjects"][1]["properties"]["data"] = function_code
transaction["frames"][0]["frameObjects"][2]["properties"]["data"] = address
transaction["frames"][0]["frameObjects"][3]["properties"]["data"] = quantity

# Response frame changes
transaction["frames"][1]["frameObjects"][1]["properties"]["data"] = function_code
transaction["frames"][1]["frameObjects"][2]["properties"]["data"] = byte_count
transaction["frames"][1]["frameObjects"][3]["properties"]["dataLength"] = byte_count


keyValues = {
    # Serial changes
    "subnetwork.properties.baudRate": changes["serial"]["baudRate"],
    "subnetwork.properties.parity": changes["serial"]["parity_enum"],
    "subnetwork.properties.stopBits": changes["serial"]["stopBits_enum"],
    "subnetwork.properties.dataBits": changes["serial"]["dataBits_enum"],

    # Node Changes
    "subnetwork.nodes[0].properties.nodeAddress": changes["node"]["nodeAddress"],
    "subnetwork.nodes[0].properties.name": changes["node"]["name"],
    "subnetwork.nodes[0].properties.modbusAddressingMode": changes["serial"]["modbusAddressingMode_enum"],

    # Transaction changes
    "subnetwork.nodes[0].transactions[0].properties.name": changes["read"]["name"],

    # Request frame changes
    "subnetwork.nodes[0].transactions[0].frames[0].frameObjects[1].properties.data": changes["read"]["function_code"],
    "subnetwork.nodes[0].transactions[0].frames[0].frameObjects[2].properties.data": changes["read"]["starting_address"],
    "subnetwork.nodes[0].transactions[0].frames[0].frameObjects[3].properties.data": changes["read"]["quantity"],

    # Response frame changes
    "subnetwork.nodes[0].transactions[0].frames[1].frameObjects[1].properties.data": changes["read"]["function_code"],
    "subnetwork.nodes[0].transactions[0].frames[1].frameObjects[2].properties.data": byte_count,
    "subnetwork.nodes[0].transactions[0].frames[1].frameObjects[3].properties.dataLength": byte_count

}
for k, v in keyValues.items():
    print(f"{k}={v}")

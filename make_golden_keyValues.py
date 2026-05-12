import json


with open("golden_configs/Modbus-RTU-ELR-3BN-IM190.json", encoding="utf-8") as f:
    gold = json.load(f)
fc = gold["subnetwork"]["nodes"][0]["transactions"][0]["frames"][0]["frameObjects"][1]["properties"]["data"]

# If byte count is present, 
def optional_length(key_values, frame_objects, frame_base):
    for i, obj in enumerate(frame_objects):
        pr = obj.get("properties", {})
        name = pr.get("name", "")

        if name == "byte_count" and "data" in pr:
            key_values[f"{frame_base}.frameObjects[{i}].properties.data"] = pr["data"]

        if name.endswith("_value") and "dataLength" in pr:
            key_values[f"{frame_base}.frameObjects[{i}].properties.dataLength"] = pr["dataLength"]

def golden_keyValues(gold):
    sp = gold["subnetwork"]["properties"]
    node = gold["subnetwork"]["nodes"][0]["properties"]
    transaction = gold["subnetwork"]["nodes"][0]["transactions"][0]

    request_frame = transaction["frames"][0]["frameObjects"]
    response_frame = transaction["frames"][1]["frameObjects"]

    # Helper function to pick the correct path for optional fields, takes obj as Frame Object, takes the base path as string, and prefer as it's the preffered keys to look for
    def pick_path(obj, base, prefer):
        # Checks the object if it has the properties field pr will be the dict. If missing, it becomes an empty dict to avoid errors.
        pr = obj.get("properties", {})
        # Iterates over the prefered list of keys
        for k in prefer:
            if k in pr and pr[k] is not None:
                # Returns the search path and the value if found.
                return f"{base}.properties.{k}", pr[k]
        # If none of the prefered keys are found, it returns the first prefered key with None value, indicating that it's missing.
        return f"{base}.properties.<missing>", None

    key_values = {
        # Serial changes
        "subnetwork.properties.physicalStandard":   sp.get("physicalStandard"),
        "subnetwork.properties.baudRate":   sp.get("baudRate"),
        "subnetwork.properties.parity": sp.get("parity"),
        "subnetwork.properties.stopBits": sp.get("stopBits"),
        "subnetwork.properties.dataBits": sp.get("dataBits"),

        # Node Changes
        "subnetwork.nodes[0].properties.nodeAddress": node.get("nodeAddress"),
        "subnetwork.nodes[0].properties.name": node.get("name"),
        "subnetwork.nodes[0].properties.modbusAddressingMode": node.get("modbusAddressingMode"),

        # Transaction changes
        "subnetwork.nodes[0].transactions[0].properties.name": transaction["properties"].get("name"),

        # Request frame changes
        "subnetwork.nodes[0].transactions[0].frames[0].frameObjects[1].properties.data": request_frame[1]["properties"].get("data"),
        "subnetwork.nodes[0].transactions[0].frames[0].frameObjects[2].properties.data": request_frame[2]["properties"].get("data"),

        #Response frame changes
        "subnetwork.nodes[0].transactions[0].frames[1].frameObjects[1].properties.data": response_frame[1]["properties"].get("data"),
        "subnetwork.nodes[0].transactions[0].frames[1].frameObjects[2].properties.data": response_frame[2]["properties"].get("data"),
    }
    # takes two variables p = path and v = value which will be returned, it looks at the fourth object in the request frame
    # in that objects properties it looks for the keys data, and if it doesn't find it it will try to look for data length instead
    # when one of the two is found, it will return the path to that key and the value.
    p, v = pick_path(
    request_frame[3],
    "subnetwork.nodes[0].transactions[0].frames[0].frameObjects[3]",
    ("data", "dataLength")
    )
    key_values[p] = v
    p, v = pick_path(
        response_frame[3],
        "subnetwork.nodes[0].transactions[0].frames[1].frameObjects[3]",
        ("dataLength", "data")
    )
    key_values[p] = v

    optional_length(
        key_values,
        request_frame,
        "subnetwork.nodes[0].transactions[0].frames[0]"
    )

    optional_length(
        key_values,
        response_frame,
        "subnetwork.nodes[0].transactions[0].frames[1]"
    )
    return key_values

key_values = golden_keyValues(gold)
for key, value in key_values.items():
    print(f"{key}={value}")    


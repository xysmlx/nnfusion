import json
import sys



assert len(sys.argv) == 3
with open(sys.argv[1], "r") as f:
    data = json.load(f)
with open(sys.argv[2], "r") as f:
    code = f.read()

data[0]["code"] = code
with open(sys.argv[1], "w") as f:
    json.dump(data, f)

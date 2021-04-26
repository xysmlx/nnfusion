import json
import sys
import os

files = os.listdir('./kernel')

def inject(jsonf, cuf):

    with open(jsonf, "r") as f:
        data = json.load(f)
    with open(cuf, "r") as f:
        code = f.read()

    data[0]["code"] = code
    with open(jsonf, "w") as f:
        json.dump(data, f)

for f in files:
    if os.path.splitext(f)[-1]=='json':
        print(f)
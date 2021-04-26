import json
import sys
import os
PREFIX = './kernel'
files = os.listdir(PREFIX)

def inject(jsonf, cuf):

    with open(jsonf, "r") as f:
        data = json.load(f)
    with open(cuf, "r") as f:
        code = f.read()

    data[0]["code"] = code
    with open(jsonf, "w") as f:
        json.dump(data, f)

for f in files:
    name = os.path.splitext(f)[0]
    suffix = os.path.splitext(f)[-1]
    if os.path.splitext(f)[-1]=='.json':
        jsonf = os.path.join(PREFIX, f)
        cuf = os.path.join(PREFIX, name+'.cu')
        assert os.path.exists(jsonf)
        assert os.path.exists(cuf)

        print(jsonf, cuf)
        inject(jsonf, cuf)
        os.system("python ../src/tools/nnfusion/kernel_db/convert_external_quantize.py %s"%jsonf)
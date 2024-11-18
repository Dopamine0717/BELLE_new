import json
import random

results=[]

with open("/work/home/acehekbmzh/lhc/xiaoqing/NLP3/1.json", "r") as f:
    temp = json.load(f)
# temp = [json.loads(l) for l in open("/work/home/acehekbmzh/lhc/xiaoqing/NLP3/excel2json-1722700456327.json", "r")]

for val in temp:
    val["review"] = val["review"].split(".")[-1]
    results.append(val)
# for val in temp1:
#     conv = [{"from": "human", "value": val['instruction']}, {"from": "assistant", "value": val['output']}]
#     results.append({'conversations': conv})

train = []
test = []
n=500
choosed=random.sample(range(len(results)), n)
for i in range(len(results)):
    if i in choosed:
        test.append(results[i])
    else:
        train.append(results[i])


with open("/work/home/acehekbmzh/lhc/xiaoqing/NLP3/train.json", "w") as f:
    for val in train:
        json.dump(val, f, ensure_ascii=False)
        f.write('\n')

with open("/work/home/acehekbmzh/lhc/xiaoqing/NLP3/val.json", "w") as f:
    for val in test:
        json.dump(val, f, ensure_ascii=False)
        f.write('\n')
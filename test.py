from copy import copy
import json
a = {
    "name": "even_",
    "config": "even_",
    "dataset": "even"
  }
res = []
for i in range(18):
    d = copy(a)
    d['name'] += str(i) + '-even'
    d['config'] += str(i)
    res.append(d)
with open('config.json','w') as f:
    json.dump(res, f)



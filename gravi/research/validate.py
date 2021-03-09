import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import legend, plot
from .data import DataReader

def calc_stat(dataset_name, mode="avg"):
  X, Y, C = DataReader.read_folder('data/' + dataset_name)
  result = [0 for i in range(len(Y[0]))]
  for y in Y:
    for i in range(len(y)):
      if y[i] != 0:
        result[i] += 1
  if mode == "avg":
    for i in range(len(result)):
      result[i] /= len(Y)
  result = np.array_split(result,C[0][0])
  result = [np.array_split(r,C[0][1]) for r in result]
  t = []
  for r in result:
    t1 = []
    for r1 in r:
      t1.append(r1.tolist())
    t.append(t1)
  result = np.array(t)
  return result

def count_net_variation(dataset_name):
  X, Y, C = DataReader.read_folder('data/' + dataset_name)
  result = {}
  for y in Y:
    k = int(''.join(['1' if abs(p) > 1e-4 else '0' for p in y]))
    if k in result:
      result[k] += 1
    else:
      result[k] = 1
  return result

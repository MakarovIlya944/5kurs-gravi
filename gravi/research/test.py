import matplotlib.pyplot as plt
import os
from .data import DataReader

def show_nets(name, params):
  x,y,c = DataReader.read_folder('data/' + name)

  fig, axs = plt.subplots(3, 3, sharex=True, sharey=True)
  for i in range(9):
    if c[i][1] == 1:
      _x = c[i][0]
      _y = c[i][2]
      m = [y[i][k*_x:(k+1)*_x] for k in range(_y)]
      j = i // 3
      k = i % 3
      axs[j, k].matshow(m,vmax=10,vmin=0)
      axs[j, k].set_title(str(i))
  plt.tight_layout()
  plt.show()

def show_3d(name):
  x,y,c = DataReader.read_folder('data/' + name)
  for i in range(len(c)):
    
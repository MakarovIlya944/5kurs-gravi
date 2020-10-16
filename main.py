
from gravi.research.data import DataReader, interpolate
import logging
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

logging.basicConfig(filename='log.txt', level=logging.WARNING)
logger = logging.getLogger('main')

def main():
  logger.info("Start")

  x = range(-10, 10)
  y = range(-10, 10)
  tr_x = []
  tr_y = []
  tr_z = []
  z = []
  for _y in y:
    for _x in x:
      v = _x*_x+_y*_y
      tr_x.append(_x)
      tr_y.append(_y)
      tr_z.append(v)
      z.append(f'[{_x}, {_y}, {v}]\n')
  with open('test.txt', 'w') as f:
    f.writelines(z)

  recs = DataReader.read_py_file("test.txt")
  
  x = range(-9, 9, 3)
  x = [_x + 0.4 for _x in x]
  y = range(-9, 9, 3)
  y = [_x + 0.4 for _x in y]
  # z = []
  # for _y in y:
  #   for _x in x:
  #     z.append(_x+_y)
  
  z = interpolate(recs, x, y)
  X, Y = np.meshgrid(x, y)
  lx = len(x)
  ly = len(y)
  Z = np.array([np.array(z[i*lx:(i+1)*lx]) for i in range(ly)])
  fig = plt.figure()
  ax = Axes3D(fig)
  ax.plot_surface(X, Y, Z, rstride=1, cstride=1)
  ax.scatter(tr_x, tr_y, tr_z, c='r')
  plt.show()


if __name__ == '__main__':
  main()

from gravi.continous.main import interpolate
from gravi.research.data import DataReader
import logging

logging.basicConfig(filename='log.txt', level=logging.WARNING)
logger = logging.getLogger('main')

def main():
  logger.info("Start")

  x = range(2, 9, 3)
  y = range(-6, -1, 2)
  z = []
  for _y in y:
    for _x in x:
      z.append(f'[{_x}, {_y}, {_x+_y}]\n')
  with open('test.txt', 'w') as f:
    f.writelines(z)

  recs = DataReader.read_py_file("test.txt")
  
  x = [i for i in range(0,2)]
  y = [i for i in range(0,2)]
  z = []
  for _y in y:
    for _x in x:
      z.append(_x+_y)
  

  X, Y, Z = interpolate(recs, x, y, [1,1])
  a = 0


if __name__ == '__main__':
  main()
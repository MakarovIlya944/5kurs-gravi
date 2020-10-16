from ..reverse.builder import *
from ..reverse.solver import Solver 
from ..continous.main import interpolate
import logging
from numpy import array,interp

class DataCreator():

  name = ''
  net_random_params = {}
  receptors_random_params = {}
  observe_points = {}
  logger = logging.getLogger('research.data.DataCreator')

  def __init__(self, params):
    super().__init__()
    self.name = params['name']
    self.net_random_params = params['net']
    self.receptors_random_params = params['receptors']
    self.observe_points = params['points']

  def create_pure_data(self, receptors):
    params = self.net_random_params
    net = center_build(params)
    s = Solver(receptors=receptors)
    return s.profile(net), net

  def intrepolate_net(self, x, y, receptors):
    # x = [0,1,2];  y = [0,3]; z = [[1,2,3], [4,5,6]]
    return interpolate(receptors, x, y)

  def create_receptors(self):
    x_r = self.receptors_random_params['x']['r']
    x_l = self.receptors_random_params['x']['l']
    x_n = self.receptors_random_params['x']['n']
    y_r = self.receptors_random_params['y']['r']
    y_l = self.receptors_random_params['y']['l']
    y_n = self.receptors_random_params['y']['n']

    x_h = (x_r - x_l) / x_n
    y_h = (y_r - y_l) / y_n
    x = [_x*x_h + x_l for _x in range(x_n)]
    y = [_y*y_h + y_l for _y in range(y_n)]
    receptors = []
    for _y in y:
      for _x in x:
        receptors.append([_x,_y,0])
    return receptors, x, y

  def create_data(self, size):
    recs, x, y = self.create_receptors()
    dGz, net = self.create_pure_data(recs)
    for i, z in enumerate(dGz):
      recs[i][2] = z
    return x, y, recs

class DataReader():
  """
  Class data-reader from diffrent sources and diffrent formats
  """

  logger = logging.getLogger('research.data.DataReader')
  
  def read_py_file(filename, x=None, y=None):
    DataReader.logger.info('read_py_file: ' + filename)
    ll = []
    with open(filename,'r') as f:
      ll = f.readlines()
    ll = [l[1:-2].split(', ') for l in ll]
    ll = [[float(n) for n in l] for l in ll]
    if not x or not y:
      return ll

    _x = [ll[0][0], ll[0][0]]
    _y = [ll[0][1], ll[0][1]]
    for l in ll:
      if l[0] < _x[0]:
        _x[0] = l[0]
      if l[0] > _x[1]:
        _x[1] = l[0]
      if l[1] < _y[0]:
        _y[0] = l[1]
      if l[1] > _y[1]:
        _y[1] = l[0]

    kx = (x[1] - x[0]) / (_x[1] - _x[0])
    dx = x[0] - _x[0]
    ky =  (y[1] - y[0]) / (_y[1] - _y[0])
    dy = y[0] - _y[0]

    ll = [[(l[0] + dx)*kx,(l[1] + dy)*ky,l[2]] for l in ll]
    return ll


def interpolate(receptors,interpolate_x,interpolate_y):

  groped_x_old, groped_y_old = {}, {}
  groped_x_new, groped_y_new = {}, {}

  for _p in receptors:
    if _p[0] in groped_y_old:
      groped_y_old[_p[0]][_p[1]] = _p[2]
    else:
      groped_y_old[_p[0]] = {_p[1]: _p[2]}
    if _p[1] in groped_x_old:
      groped_x_old[_p[1]][_p[0]] = _p[2]
    else:
      groped_x_old[_p[1]] = {_p[0]: _p[2]}

  for _y in groped_x_old:
    points = []
    values = []
    for _x in groped_x_old[_y]:
      points.append(_x)
      values.append(groped_x_old[_y][_x])
    groped_x_new[_y] = interp(interpolate_x,points, values)

  for _x in groped_y_old:
    points = []
    values = []
    for _y in groped_y_old[_x]:
      points.append(_y)
      values.append(groped_y_old[_x][_y])
    groped_y_new[_x] = interp(interpolate_y,points, values)

  z = []
  old_x = [x for x in groped_y_new]
  old_y = [y for y in groped_x_new]
  sorted(old_x)
  sorted(old_y)
  rng_x = range(len(old_x) - 1)
  rng_y = range(len(old_y) - 1)

  for i_y, _y in enumerate(interpolate_y):
    for i_x, _x in enumerate(interpolate_x):
      for i in rng_x:
        if old_x[i] <= _x and old_x[i+1] >= _x:
          Y = groped_y_new[old_x[i]][i_y] + (groped_y_new[old_x[i+1]][i_y] - groped_y_new[old_x[i]][i_y]) * (_x - old_x[i]) / (old_x[i+1] - old_x[i])
          break
      for i in rng_y:
        if old_y[i] <= _y and old_y[i+1] >= _y:
          X = groped_x_new[old_y[i]][i_x] + (groped_x_new[old_y[i+1]][i_x] - groped_x_new[old_y[i]][i_x]) * (_y - old_y[i]) / (old_y[i+1] - old_y[i])
          break
      z.append((X + Y) / 2.0)
  return z

def test():
  params = {
    'name': 'test',
    'net': {
      'count': (5,1,5),
      'right': (3000,50,-1500),
      'left': (1000,0,-500),
      'width': (1,0,1),
      'center': (1,0,1),
      'c_value': 1,
      },
    'receptors':{
      'x':{
        'r': 3000,
        'l': 1000,
        'n': 10
      },
      'y':{
        'r': 0,
        'l': -2000,
        'n': 10
      }
    },
    'points':
    []
  }
  size = 100
  d = DataCreator(params)
  x, y, recs = d.create_data(size)

  with open('dgz.txt', 'w') as f:
    f.writelines([str(r) + '\n' for r in recs])

  # dGzInterpolate = d.intrepolate_net(x, y, recs, dGz)
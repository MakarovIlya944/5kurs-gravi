from os import name
from ..reverse.builder import *
from ..reverse.solver import Solver 
from numpy import array,interp
from torch import Tensor
from random import *
import os
from config import get_logger, log_config
import json

class DataCreator():

  name = ''
  net_random_params = {}
  receptors_random_params = {}
  observe_points = {}
  logger = get_logger(__name__ + '.DataCreator')

  def __init__(self, params):
    super().__init__()
    seed()
    self.name = params['name']
    self.net_random_params = params['net']
    self.receptors_random_params = params['receptors']
    self.observe_points = params.get('points')

  def create_pure_data(self, receptors):
    net = center_build(self.randomize())
    s = Solver(receptors=receptors)
    return s.profile(net), net

  def randomize(self):
    params = dict(self.net_random_params)
    n = params['count']
    p = params.get('center')
    max_p = tuple([k-1 for k in n])
    min_p = (0,0,0)
    if p.get('max'):
      max_p = tuple([min(max_p[i], p['max'][i]) for i in range(3)])
    if p.get('min'):
      min_p = tuple([max(min_p[i], p['min'][i]) for i in range(3)])
    params['center'] = tuple([randint(min_p[i], max_p[i]) for i in range(3)])
    self.logger.debug('center:' + str(params['center']))

    p = params.get('width')
    max_p = tuple([k-1 for k in n])
    min_p = (0,0,0)
    if p.get('max'):
      max_p = tuple([min(max_p[i], p['max'][i]) for i in range(3)])
    if p.get('min'):
      min_p = tuple([max(min_p[i], p['min'][i]) for i in range(3)])
    params['width'] = tuple([randint(min_p[i], max_p[i]) for i in range(3)])
    self.logger.debug('width:' + str(params['width']))

    params['c_value'] = random() * params['c_value']
    return params

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

  def create_data(self):
    recs, x, y = self.create_receptors()
    dGz, net = self.create_pure_data(recs)
    for i, z in enumerate(dGz):
      recs[i][2] = z
    return x, y, recs,net
    
  def read_dataset(self, n):
    if self.observe_points:
      is_interpolated = True
      observe_x = self.observe_points[0]
      observe_y = self.observe_points[1]
    else:
      is_interpolated = False
  
    self.logger.info(('creating interpolated dataset' if is_interpolated else 'creating dataset') + name)
    log_step = int(n * log_config['data_creation'])
    
    for i in range(n):
      
      filename= os.path.abspath('.') + f'/data/{self.name}/{i}'
      x, y, r,net = self.create_data()
      if is_interpolated:
        z = interpolate(r, observe_x, observe_y)
      else:
        z = [el[2] for el in r]
      with open(filename+'_in', 'w') as f:
        j = 0
        for _y in y:
          for _x in x:
            f.write(str(z[j]) + '\n')
            j += 1
      with open(filename + '_out', 'w') as f:
        for p in net:
          f.write(str(p[1]) + '\n')
      with open(filename + '_out_config', 'w') as f:        
        f.write(' '.join([str(k) for k in net.n]))
      if not n % log_step:
        self.logger.info(f'set #{i} created')
    return len(z), len(net)

class DataReader():
  """
  Class data-reader from diffrent sources and diffrent formats
  """

  logger = get_logger(__name__ + '.DataReader')
  
  def read_py_file(filename, x=None, y=None):
    DataReader.logger.info('reading_py_file: ' + filename)
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


  """
  Read dataset from <path> folder.
  Return:
  X - input data: z-value of receptors ordered by y,x
  Y - output data: solidity of net cells, ordered by y,x
  C - net dimensions
  """
  def read_folder(path, out_format='default'):
    i = 0
    filename = path + f'/{i}'
    X, Y, C = [], [], []
    while os.path.exists(filename + '_in'):
      with open(filename + '_in', 'r') as f:
        ll = f.readlines()
      X.append([float(l) for l in ll])
      with open(filename + '_out', 'r') as f:
        ll = f.readlines()
      Y.append([float(l) for l in ll])
      with open(filename + '_out_config', 'r') as f:
        ll = f.readlines()[0].split(' ')
      C.append([int(l) for l in ll])
      i += 1
      filename = path + f'/{i}'
    DataReader.logger.info(f'X: {len(X)}x{len(X[0])} Y: {len(Y)}x{len(Y[0])} C: {len(C)}x{len(C[0])}')
    if out_format == 'default':
      return X, Y, C
    elif out_format == 'tensor':
      return Tensor(array(X)), Tensor(array(Y)), Tensor(array(C))
    else:
      DataReader.logger.error('Unexpected out format type')
      raise KeyError('Unexpected out format type')

class Configurator():
  """
  Class reader configs for datasets and models
  """

  logger = get_logger(__name__ + '.Configurator')

  def get_dataset_config(name):
    return Configurator.__read_file(name, 'dataset')

  def get_model_config(name):
    return Configurator.__read_file(name, 'model')

  def __read_file(name, file_type):
    path = os.path.abspath('.') + f'/configs/{file_type}/{name}.json'
    if not os.path.exists(path):
      Configurator.logger.error(f'{file_type} config {name} not exist')
      raise FileNotFoundError(f'{file_type} config {name} not exist')
    with open(path, 'rb') as f:
      return json.load(f)

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

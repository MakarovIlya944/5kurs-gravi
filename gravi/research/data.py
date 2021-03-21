from copy import copy
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
    self.create_mode = params.get('create_mode')

  def create_pure_data(self, receptors, build_params=None):
    if not build_params:
      build_params = self.randomize()
    if self.create_mode == "fill":
      net = line_build(build_params)
    elif self.create_mode == 'circle':
      net = circle_build(build_params)
    else:
      net = center_build(build_params)
    s = Solver(receptors=receptors,net=net)
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

  def create_fill_params(self):
    params = dict(self.net_random_params)

    n = params['count']
    bodies_per_line = params.get('bodies_per_line')
    if not bodies_per_line:
      raise KeyError(f"Config havent bodies_per_line config: {str(params)}")
    if n[1] != 1:
      raise KeyError(f"Net not flat: {str(n)}")

    nets = []
    # TODO fix 3d nets (not flat)
    for i in range(n[2]):
      tmp = [n[0]]
      for j in range(bodies_per_line):
        m = max(tmp)
        d = randint(1, tmp.pop(tmp.index(m))-1)
        tmp.append(d)
        tmp.append(m - d)
      DataCreator.logger.debug(tmp)
      nets.append(tmp)

    result = []
    params['c_value'] = random() * params['c_value']
    for i, raw in enumerate(nets):
      offset = 0
      for line in raw:
        setting = copy(params)
        setting['line_begin'] = [offset, 0, i]
        setting['length'] = line
        offset += line
        result.append(setting)

    return result

  def create_data(self, receptors, x, y):
    result = []
    if self.create_mode == "fill":
      params = self.create_fill_params()
      for p in params:
        recs = [copy(r) for r in receptors]
        dGz, net = self.create_pure_data(recs, build_params=p)
        for i, z in enumerate(dGz):
          recs[i][2] = z
        result.append((x, y, recs, net))
    else:
      dGz, net = self.create_pure_data(receptors)
      for i, z in enumerate(dGz):
        receptors[i][2] = z
      result.append((x, y, receptors, net))
    return result

  def read_dataset(self, n):
    if self.observe_points:
      is_interpolated = True
      observe_x = self.observe_points[0]
      observe_y = self.observe_points[1]
    else:
      is_interpolated = False
  
    self.logger.info(('creating interpolated dataset' if is_interpolated else 'creating dataset') + name)
    log_step = int(n * log_config['data_creation'])
    if log_step == 0:
      log_step = 1
    
    i = 0
    while i < n:
      recs, x, y = self.create_receptors()
      data = self.create_data(copy(recs), x, y)
      for j, (x, y, r, net) in enumerate(data):
        filename= os.path.abspath('.') + f'/data/{self.name}/'
        if is_interpolated:
          z = interpolate(r, observe_x, observe_y)
        else:
          z = [el[2] for el in r]
        with open(filename+f'{i}_in', 'w') as f:
          l = 0
          for _y in y:
            for _x in x:
              f.write(str(z[l]) + '\n')
              l += 1
        with open(filename + f'{i}_out', 'w') as f:
          for p in net:
            f.write(str(p[1]) + '\n')
        with open(filename + f'{i}_out_config', 'w') as f:
          f.write(' '.join([str(k) for k in net.n]))
        if not n % log_step:
          self.logger.info(f'set #{i} created')
        i += 1
    return len(z), len(net)

  def save_predicted(predicted_data):
    DataCreator.logger.info(f'Save predicted data {predicted_data["name"]}')
    filename= os.path.abspath('.') + f'/result/{predicted_data["name"]}.json'
    data = predicted_data['data']
    for d in data:
      d['l2_diff'] = d['l2_diff'].tolist()
    with open(filename, 'w') as f:
      f.write(json.dumps(data))

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
  Read dataset from <path>.
  Return:
  X - input data: z-value of receptors ordered by y,x
  Y - output data: solidity of net cells, ordered by y,x
  C - net dimensions
  """
  def read_one(path, index, out_format='default', shape='default'):
    DataReader.log_step = log_config['data_read']
    filename = path + f'/{index}'
    X, Y, C = [], [], []
    with open(filename + '_in', 'r') as f:
      ll = f.readlines()
    X.append([float(l) for l in ll])
    with open(filename + '_out', 'r') as f:
      ll = f.readlines()
    Y.append([float(l) for l in ll])
    with open(filename + '_out_config', 'r') as f:
      ll = f.readlines()[0].split(' ')
    C.append([int(l) for l in ll])
    DataReader.logger.debug(f"Read {index} data")
    msg = f'X: {len(X)}x{len(X[0])} Y: {len(Y)}x{len(Y[0])} C: {len(C)}x{len(C[0])}'
    if shape != 'default':
      shape = tuple([len(C)] + shape)
      shape_out = tuple([len(C), C[0][2], C[0][0], C[0][1]])
      X = array(X).reshape(shape)
      Y = array(Y).reshape(shape_out)
      shape = 'x'.join([str(s) for s in shape])
      shape_out = 'x'.join([str(s) for s in shape_out])
      msg = f'X: {shape} Y: {shape_out} C: {len(C)}x{len(C[0])}'
    DataReader.logger.info(msg)
    if out_format == 'default':
      return X, Y, C
    elif out_format == 'tensor':
      return Tensor(array(X)), Tensor(array(Y)), Tensor(array(C))
    else:
      DataReader.logger.error('Unexpected out format type')
      raise KeyError('Unexpected out format type')

  """
  Read dataset from <path> folder.
  Return:
  X - input data: z-value of receptors ordered by y,x
  Y - output data: solidity of net cells, ordered by y,x
  C - net dimensions
  """
  def read_folder(path, out_format='default', shape='default'):
    i = 0
    DataReader.log_step = log_config['data_read']
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
      if i % DataReader.log_step == DataReader.log_step - 1:
        DataReader.logger.debug(f"Read {i} data")
    msg = f'X: {len(X)}x{len(X[0])} Y: {len(Y)}x{len(Y[0])} C: {len(C)}x{len(C[0])}'
    if shape != 'default':
      shape = tuple([len(C)] + shape)
      shape_out = tuple([len(C), C[0][2], C[0][0], C[0][1]])
      X = array(X).reshape(shape)
      Y = array(Y).reshape(shape_out)
      shape = 'x'.join([str(s) for s in shape])
      shape_out = 'x'.join([str(s) for s in shape_out])
      msg = f'X: {shape} Y: {shape_out} C: {len(C)}x{len(C[0])}'
    DataReader.logger.info(msg)
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
  pytorch_train_eps = 5e-2

  def get_dataset_config(name):
    return Configurator.__read_file(name, 'dataset')

  def get_model_config(name):
    return Configurator.__read_file(name, 'model')

  def get_predict_config(name):
    return Configurator.__read_file(name, 'predict')

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

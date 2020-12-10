from gravi.reverse.builder import complex_build
from numpy.core.records import array
from gravi.reverse import net,min
from .data import DataCreator, DataReader, Configurator
from datetime import datetime
from .models.pytorch import ModelPyTorch
import numpy.linalg as l
import numpy as np
import os
from copy import copy
from config import get_logger

logger = get_logger(__name__)

def research(size, dataset_name):
  i,o = prepare_data(size, dataset_name)
  learn(i, o, dataset_name)

def learn(len_i, len_o, dataset_name, model_config_name, model_params=None):
  shape = 'default'
  if not model_params:
    model_params = Configurator.get_model_config(model_config_name)
    if model_params.get('type') and model_params['type'] == "cnn":
      shape = [1,model_params['shape']['in']['w'],model_params['shape']['in']['h']]
    else:
      tmp = model_params['layers'][0]['w']
      if tmp != len_i:
        logger.error(f'input dim in config {tmp} not equal {len_i}')
        raise AssertionError(f'input dim in config {tmp} not equal {len_i}')
      tmp = model_params['layers'][-1]['w']
      if tmp != len_o:
        logger.error(f'output dim in config {tmp} not equal {len_o}')
        raise AssertionError(f'output dim in config {tmp} not equal {len_o}')
  model_params['model_config_name'] = model_config_name
  base_path = os.path.abspath('.') +  f'/models/'
  date = '/' + datetime.now().strftime('%m-%d-%H-%M') + '_' + model_config_name + '-' + dataset_name

  mp = ModelPyTorch(model_params)
  logger.info("ModelPyTorch model begin learn")
  X, Y, C = DataReader.read_folder('data/' + dataset_name, out_format='tensor',shape=shape)
  try:
    mp.learn(X, Y)
    logger.info("ModelPyTorch model end learn")
  except KeyboardInterrupt:
    logger.info("ModelPyTorch model learning interrupted")
  finally:
    mp.save(base_path + mp.name + date)
    logger.info("ModelPyTorch model saved")

def prepare_data(size, name, config_name, params=None):
  if not params:
    params = Configurator.get_dataset_config(config_name)
  params['name'] = name
  path = os.path.abspath('.') + f'/data/{name}'
  try:
    os.mkdir(path) 
  except FileExistsError:
    logger.warn(f'dir for dataset {name} already exist')
    pass
  
  d = DataCreator(params)
  len_i, len_o = d.read_dataset(size)
  return len_i, len_o

def predict(predict_name, is_save=False, net_index=None, model_index=None):
  predict_params = Configurator.get_predict_config(predict_name)
  dataset_name = predict_params[0]['dataset']
  for d in predict_params:
    if d['dataset'] != dataset_name:
      raise AssertionError('Diffrent datasets not implemented')
  logger.debug("Begin to read")

  shape = 'default'
  model_params = Configurator.get_model_config(predict_params[0]['config'])
  if model_params.get('type') and model_params['type'] == 'cnn':
    shape = [1,model_params['shape']['w'],model_params['shape']['h']]
    logger.debug("Change dataset shape")

  if net_index is None:
    X, Y, C = DataReader.read_folder('data/' + dataset_name, out_format='tensor', shape=shape)
  else:
    X, Y, C = DataReader.read_one('data/' + dataset_name, net_index, out_format='tensor', shape=shape)

  Y = Y.detach().numpy()
  shape = Y.shape[1:]
  k = 1
  for e in shape:
    k *= e
  shape = (Y.shape[0], k)

  if model_index is None:
    for d in predict_params:
      predict_one(d,X,Y,shape)
  else:
    predict_one(predict_params[model_index],X,Y,shape)

  predicted = {'name':predict_name, 'data': predict_params }
  if is_save:
    DataCreator.save_predicted(predicted)
  if shape == 'default':
    return predicted, X.detach().numpy(), Y, C.detach().numpy().astype(int)
  else:
    return predicted, X.detach().numpy(), Y.reshape(shape), C.detach().numpy().astype(int)

def predict_one(d,X,Y,shape):
  model_params = Configurator.get_model_config(d['config'])
  model_params['model_config_name'] = d['name']
  mp = ModelPyTorch(model_params, True)
  path = os.path.abspath('.') + '/models/pytorch/' + d['name']
  mp.load(path)
  logger.info(f"ModelPyTorch model {d['name']} begin predict")
  _Y = mp.predict(X).detach().numpy()
  d['l2_diff'] = l.norm(Y - _Y)
  _Y = _Y.reshape(shape)
  d['predicted'] = _Y
  logger.info(f"ModelPyTorch model {d['name']} end predict")

def inspect(dataset_name, command, dataset_config=None, index=0, model_name=False, model_config=False):
  if command == 'stat':
    return {}, calc_stat(dataset_name)
  elif command == 'response' or command == 'reverse' or command == 'reverse-net':
    dataset_config = Configurator.get_dataset_config(dataset_config)
    s = dataset_config['net']['count']
    def_net = dataset_config['net']
    correct = copy(def_net)
    correct['values'] = {}
    if model_name:
      X,Y,C = DataReader.read_one('data/' + dataset_name, index, out_format='tensor')
      d = {'config':model_config, 'name':model_name}
      predict_one(d,X,Y,s)
      Y = d['predicted']
      for i in range(len(Y)):
        for j in range(len(Y[i])):
          for k,v in enumerate(Y[i][j]):
            correct['values'][(i,j,k)] = v
    else:
      X,Y,C = DataReader.read_one('data/' + dataset_name, index)
      for i,v in enumerate(Y[0]):
        correct['values'][(i%s[0],(i%(s[0]*s[1]))//s[0],i//(s[0]*s[1]))] = v
    correct = complex_build(params = correct)
    def_net["default"] = 0.1
    net = complex_build(params = def_net)
    r_x = dataset_config['receptors']['x']
    r_y = dataset_config['receptors']['y']
    r = (dataset_config['receptors']['x']['n'], dataset_config['receptors']['y']['n'])
    r_x = range(r_x['l'],r_x['r'],(r_x['r'] - r_x['l']) // r_x['n'])
    r_y = range(r_y['l'],r_y['r'],(r_y['r'] - r_y['l']) // r_y['n'])
    receptors = []
    for y in r_y:
      for x in r_x:
        receptors.append([float(x),float(y),0.0])
    receptors = np.asarray(receptors)
    alpha=[0]
    gamma=None
    smile = min.Minimizator(net=net, receptors=receptors, correct=correct, alpha=alpha, gamma=gamma, dryrun=True)
    if command == 'response':
      return {'r_x': r_x, 'r_y':r_y}, np.asarray(smile.solver.dGz).reshape(r)
    net = smile.minimization()
    dGz = smile.solver.profile(net)
    if command == 'reverse':
      return {'r_x': r_x, 'r_y':r_y}, np.asarray(dGz).reshape(r)
    if command == 'reverse-net':
      n = dataset_config['net']
      if n['count'][1] != 1:
        raise Exception('Net not thin')
      r_x = range(n['left'][0],n['right'][0],(n['right'][0] - n['left'][0]) // n['count'][0])
      r_y = range(n['left'][2],n['right'][2],(n['right'][2] - n['left'][2]) // n['count'][2])
      logger.debug(f'Net shape by config: {s}')
      logger.debug(f'Net shape by reverse: {net.n}')
      return {'r_x': r_x, 'r_y':r_y}, net.asarray()

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
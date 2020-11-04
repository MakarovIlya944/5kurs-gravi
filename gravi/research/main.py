from .data import DataCreator, DataReader, Configurator
from datetime import datetime
from .models.pytorch import ModelPyTorch
import os
from config import get_logger

logger = get_logger(__name__)

def research(size, dataset_name):
  i,o = prepare_data(size, dataset_name)
  learn(i, o, dataset_name)

def learn(len_i, len_o, dataset_name, model_config_name, model_params=None):
  if not model_params:
    model_params = Configurator.get_model_config(model_config_name)
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
  X, Y, C = DataReader.read_folder('data/' + dataset_name, out_format='tensor')
  mp.learn(X, Y)
  logger.info("ModelPyTorch model end learn")
  mp.save(base_path + mp.name + date)

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
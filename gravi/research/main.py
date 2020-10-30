from .data import DataCreator, DataReader
from datetime import datetime
from .models.pytorch import ModelPyTorch
from .models.spider import Spider
import os
from config import get_logger,prepare_data_default_params,model_learn_default_params

logger = get_logger(__name__)

def research(size, dataset_name):
  i,o = prepare_data(size, dataset_name)
  learn(i, o, dataset_name)

def learn(len_i, len_o, dataset_name, model_params=None):
  if not model_params:
    model_params = model_learn_default_params
    model_params['layers'] = [
        {'w': len_i},
        {'w': int((len_i+len_o)/2)},
        {'w': len_o},
      ]
  base_path = os.path.abspath('.') +  f'/models/'
  date = '/' + datetime.now().strftime('%m-%d-%h-%M')

  mp = ModelPyTorch(model_params)
  logger.warn("ModelPyTorch model begin learn")
  X, Y = DataReader.read_folder('data/' + dataset_name)
  mp.learn(X, Y)
  logger.warn("ModelPyTorch model end learn")
  mp.save(base_path + mp.name + date)

  model_params['save_path'] = base_path + Spider.name + date
  ms = Spider(model_params)
  logger.warn("Spider model begin learn")
  ms.learn('data/' + dataset_name, save=True)
  logger.warn("Spider model end learn")

def prepare_data(size, name, params=None):
  if not params:
    params = prepare_data_default_params
    params['name'] = name
  path = os.path.abspath('.') + f'/data/{name}'
  try:
    os.mkdir(path) 
  except FileExistsError:
    logger.warn(f'dir for dataset {name} already exist')
    pass
  
  d = DataCreator(params)
  len_i, len_o = d.create_dataset(size)
  return len_i, len_o
from .data import DataCreator, DataReader
from datetime import datetime
from .models.pytorch import ModelPyTorch
import logging
logger = logging.getLogger(__name__)

def research(size,dataset):
  logger.info("researching")
  i,o = prepare_data(size)
  learn(i,o,dataset)

def learn(len_i, len_o, dataset):
  model_params = {
    'iters': 1000,
    'layers':[
      {'w': len_i},
      {'w': int((len_i+len_o)/2)},
      {'w': len_o},
    ],
    'lr':0.01
  }
  mp = ModelPyTorch(model_params)
  X, Y = DataReader.read_folder('data/' + dataset)
  mp.learn(X,Y)
  mp.save('models/' + mp.name + '/models' + datetime.now().strftime('%m-%d-%h-%M'))

def prepare_data(size):
  logger.info('prepare_data')
  params = {
    'name': 'test',
    'net': {
      'count': (5,1,5),
      'right': (3000,50,-1500),
      'left': (1000,0,-500),
      # 'width': (1,0,1),
      # 'center': (1,0,1),
      'c_value': 10,
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
    }
  }
  d = DataCreator(params)
  len_i, len_o = d.create_dataset(size)
  return len_i, len_o
from gravi.research.data import Configurator
import torch
from config import get_logger
from .pytorch import ModelPyTorch, CNN_Net, Net
from torch.nn import MSELoss
from config import get_logger, log_config
from torch.optim import SGD
from torch import split,load,save

logger = ''

class ModelPyTorchMany(ModelPyTorch):

  models = []
  prev_loss = 1e+100
  lr = None

  def __init__(self, params, dataset_config, is_predict=False):
    super().__init__()
    global logger
    logger = get_logger(__name__, params['model_config_name'] + '.log')
    n = Configurator.get_dataset_config(dataset_config)['net']['count']
    if n[1] != 1:
      raise Exception('Net not flat')
    if params.get('type') and params['type'] == "cnn":
      shape_out = (params['shape']['out']['z'],params['shape']['out']['x'],params['shape']['out']['y'])
      for i in range(n[0]):
        tmp = []
        for j in range(n[2]):
          tmp.append(CNN_Net(params['layers'], shape_out=shape_out))
        self.models.append(tmp)
    else:
      for i in range(n[0]):
        tmp = []
        for j in range(n[2]):
          tmp.append(Net(params['layers']))
        self.models.append(tmp)
    if not is_predict:
      if params.get('runAllIters'):
        self.allIters = params['runAllIters']
      else:
        self.allIters = True
      self.log_step = log_config['pytorch']
      self.criterion = MSELoss()
      if type(params['lr']) == type(1.1):
        self.optimizer = SGD(self.models.parameters(), lr=params['lr'])
      elif type(params['lr']) == type({}):
        self.lr = params['lr']
      self.iteraions = params['iters']
      self.trainDatsetPart = params['trainDatasetPart']

  def learn(self, x, y):
    self.log_step *= self.iteraions
    l = len(x)
    k = self.trainDatsetPart
    divideDataset = [int(l*k), l - int(l*k)]
    l = split(x, divideDataset)
    train_x = l[0]
    val_x = l[1]
    l = split(y, divideDataset)
    train_y = l[0]
    val_y = l[1]
    if self.lr:
      lr = self.lr['default']
      dlr = self.lr['decrease']
      nlr = self.lr['detente']
      ilr = 0
    for t in range(self.iteraions):
      y_pred_train = self.model(train_x)
      loss_train = self.criterion(y_pred_train, train_y)

      y_pred_val = self.model(val_x)
      loss_val = self.criterion(y_pred_val, val_y)

      if not self.allIters and (loss_val.item() - self.prev_loss) > Configurator.pytorch_train_eps:
        logger.info(f'#{t} loss train:{loss_train.item()} val:{loss_val.item()}')
        break
      self.prev_loss = loss_val.item()
      if t % self.log_step == self.log_step - 1:
        logger.info(f'#{t} loss train:{loss_train.item()} val:{loss_val.item()}')
      else:
        logger.debug(f'#{t} loss train:{loss_train.item()} val:{loss_val.item()}')

      # Zero gradients, perform a backward pass, and update the weights.
      if self.lr:
        self.optimizer = SGD(self.model.parameters(), lr=lr)
        lr *= dlr
        ilr += 1
        if ilr > nlr:
          ilr = 0
          lr = self.lr['default']
      self.optimizer.zero_grad()
      loss_train.backward()
      self.optimizer.step()

  def predict(self, x):
    return self.model(x)

  def save(self, path):
    save(self.model.state_dict(), path)

  def load(self, path):
    self.model.load_state_dict(load(path))
    self.model.eval()
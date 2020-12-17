from gravi.research.data import Configurator
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from config import get_logger, log_config

logger = ''

class Net(nn.Module):
  def __init__(self, layers):
    super(Net, self).__init__()
    self._layers = []
    self.relu = nn.ReLU()
    rng = range(len(layers)-1)
    for i in rng:
      self._layers.append(nn.Linear(layers[i]['w'], layers[i+1]['w']))
    self.layers = nn.ModuleList(self._layers)

  def forward(self, x):
    for l in self.layers:
      x = l(x)
      x = self.relu(x)
    return x

class CNN_Net(nn.Module):
  """
  w = (w - k + 2p) / s + 1
  """
  default_cnn = {
    'conv':{
      'k':5,
      's':1,
      'p':2
    },
    'pool':{
      'k':2,
      's':2
    }
  }

  def __init__(self, layers):
    super(CNN_Net, self).__init__()

    self._layers = []
    self.relu = nn.ReLU()
    rng = range(len(layers)-1)
    reshape = -1
    for i in rng:
      if layers[i].get('type') and layers[i].get('type') == "cnn":
        layer = self._conv_layer_set(layers[i]['in'], layers[i]['out'], layers[i])
      elif layers[i].get('type') and layers[i].get('type') == "drop":
        layer = nn.Dropout()
      elif layers[i].get('type') and layers[i].get('type') == "reshape":
        if layers[i].get('w') and layers[i].get('h'):
          w = layers[i].get('w')
          h = layers[i].get('h')
          d = layers[i].get('d')
          _layer = lambda x: torch.reshape(x, (w,h,d))
        else:
          _layer = lambda x: torch.reshape(x, (-1,))
        reshape = {i: _layer}
      else:
        layer = nn.Linear(layers[i]['w'], layers[i+1]['w'])
      try:
        self._layers.append(layer)
      except Exception:
        pass
    i = len(layers)-1
    if layers[i].get('type') and layers[i].get('type') == "cnn":
      layer = self._conv_layer_set(layers[i]['in'], layers[i]['out'], layers[i])
      self._layers.append(layer)
    self.reshape = reshape
    self.layers = nn.ModuleList(self._layers)

  def _conv_layer_set(self, in_c, out_c, params):
    conv = params['conv']
    pool = params['pool']
    conv_layer = nn.Sequential( 
      nn.Conv2d(in_c, out_c, kernel_size=conv['k'], stride=conv['s'], padding=conv['p']), 
      nn.ReLU(), 
      nn.MaxPool2d(kernel_size=pool['k'], stride=pool['s'], padding=pool['p'])
    ) 
    return conv_layer
  
  def forward(self, x):
    if len(self.reshape) != 0:
      n = list(self.reshape)[0]
      for i in range(n):
        x = self.layers[i](x)
      x = self.reshape[n](x)
      for i in range(n+1,len(self.layers)):
        x = self.layers[i](x)
    else:
      for i in range(len(self.layers)):
        x = self.layers[i](x)
    return x

class ModelPyTorch():
  name = "pytorch"
  prev_loss = 1e+100

  def __init__(self, params, is_predict=False):
    super().__init__()
    global logger
    logger = get_logger(__name__, params['model_config_name'] + '.log')
    if params.get('type') and params['type'] == "cnn":
      self.model = CNN_Net(params['layers'])
    else:
      self.model = Net(params['layers'])
    if not is_predict:
      if params.get('runAllIters'):
        self.allIters = params['runAllIters']
      else:
        self.allIters = True
      self.log_step = log_config['pytorch']
      self.criterion = nn.MSELoss()
      self.optimizer = optim.SGD(self.model.parameters(), lr=params['lr'])
      self.iteraions = params['iters']
      self.trainDatsetPart = params['trainDatasetPart']

  def learn(self, x, y):
    self.log_step *= self.iteraions
    l = len(x)
    k = self.trainDatsetPart
    divideDataset = [int(l*k), l - int(l*k)]
    l = torch.split(x, divideDataset)
    train_x = l[0]
    val_x = l[1]
    l = torch.split(y, divideDataset)
    train_y = l[0]
    val_y = l[1]
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
      self.optimizer.zero_grad()
      loss_train.backward()
      self.optimizer.step()

  def predict(self, x):
    return self.model(x)

  def save(self, path):
    torch.save(self.model.state_dict(), path)

  def load(self, path):
    self.model.load_state_dict(torch.load(path))
    self.model.eval()
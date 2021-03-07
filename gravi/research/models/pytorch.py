from numpy.core import shape_base
from gravi.research.data import Configurator
import torch
from json import dumps
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from config import get_logger, log_config

logger = ''

class Net(nn.Module):
  def __init__(self, layers):
    super(Net, self).__init__()
    self._layers = []
    rng = range(len(layers)-1)
    for i in rng:
      try:
        if layers[i].get('type') == "relu":
          self._layers.append(nn.ReLU())
        elif layers[i].get('type') == "tanh":
          self._layers.append(nn.Tanh())
        elif layers[i].get('type') == "sigmoid":
          self._layers.append(nn.Sigmoid())
        elif layers[i].get('type') == "lrelu":
          self._layers.append(nn.LeakyReLU())
        else:
          self._layers.append(nn.Linear(layers[i]['w'], layers[i+1]['w']))
      except KeyError as ex:
        logger.error(f'i: {i} {dumps(layers[i])}')
        logger.error(f'i+1: {i+1} {dumps(layers[i+1])}')
        raise ex
    self.layers = nn.ModuleList(self._layers)

  def forward(self, x):
    for l in self.layers:
      x = l(x)
    return x

def reshape_final(x, shape):
  return x.view(tuple([x.size()[0]] + list(shape)))

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

  def __init__(self, layers, shape_out=None):
    super(CNN_Net, self).__init__()

    self._layers = []
    rng = range(len(layers)-1)
    reshape = -1
    for i in rng:
      if not layers[i].get('type'):
        continue
      if layers[i].get('type') == "cnn":
        layer = self._conv_layer_set(layers[i]['in'], layers[i]['out'], layers[i])
      elif layers[i].get('type') == "drop":
        layer = nn.Dropout(p=0.3)
      elif layers[i].get('type') == "reshape":
        if layers[i].get('x') and layers[i].get('y') and layers[i].get('z'):
          _x = layers[i].get('x')
          y = layers[i].get('y')
          z = layers[i].get('z')
          _layer = lambda x: x.view((_x,y,z))
        else:
          _layer = lambda x: x.view(-1, self.num_flat_features(x))
        reshape = {i: _layer}
        layer = nn.Linear(layers[i]['w'], layers[i+1]['w'])
      else:
        layer = nn.Linear(layers[i]['w'], layers[i+1]['w'])
      self._layers.append(layer)
    i = len(layers)-1
    if layers[i].get('type') and layers[i].get('type') == "cnn":
      layer = self._conv_layer_set(layers[i]['in'], layers[i]['out'], layers[i])
      self._layers.append(layer)
    if shape_out:
      reshape[len(layers)] = lambda x: reshape_final(x, shape_out)
    self.reshape = reshape
    self.layers = nn.ModuleList(self._layers)

  def num_flat_features(self, x):
      size = x.size()[1:]  # all dimensions except the batch dimension
      num_features = 1
      for s in size:
          num_features *= s
      return num_features

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
    rng = len(self.reshape) + len(self.layers)
    rng = range(rng)
    offset = 0
    # logger.debug(f'data init shape: {x.shape}')
    for i in rng:
      if i in self.reshape:
        x = self.reshape[i](x)
        offset += 1
      else:
        x = self.layers[i - offset](x)
      # logger.debug(f'data shape: {x.shape}')
    return x

class ModelPyTorch():
  name = "pytorch"
  prev_loss = 1e+100
  lr = None

  def __init__(self, params, is_predict=False):
    super().__init__()
    global logger
    logger = get_logger(__name__, params['model_config_name'] + '.log')
    if params.get('type') and params['type'] == "cnn":
      shape_out = (params['shape']['out']['z'],params['shape']['out']['x'],params['shape']['out']['y'])
      self.model = CNN_Net(params['layers'], shape_out=shape_out)
    else:
      self.model = Net(params['layers'])
    if not is_predict:
      if params.get('runAllIters'):
        self.allIters = params['runAllIters']
      else:
        self.allIters = True
      self.log_step = log_config['pytorch']
      self.criterion = nn.MSELoss()
      if type(params['lr']) == type(1.1):
        self.optimizer = optim.SGD(self.model.parameters(), lr=params['lr'])
      elif type(params['lr']) == type({}):
        self.lr = params['lr']
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
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
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
    torch.save(self.model.state_dict(), path)

  def load(self, path):
    self.model.load_state_dict(torch.load(path))
    self.model.eval()
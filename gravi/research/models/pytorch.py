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

class ModelPyTorch():
  name = "pytorch"
  prev_loss = 1e+100

  def __init__(self, params, is_predict=False):
    super().__init__()
    global logger
    logger = get_logger(__name__, params['model_config_name'] + '.log')
    self.model = Net(params['layers'])
    if not is_predict:
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

      if (loss_val.item() - self.prev_loss) > Configurator.pytorch_train_eps:
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
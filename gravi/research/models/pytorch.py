import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import datetime
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

  def __init__(self, params):
    super().__init__()
    self.log_step = log_config['pytorch']
    self.model = Net(params['layers'])
    global logger
    logger = get_logger(__name__,params['model_config_name']+ '.log')
    self.criterion = nn.MSELoss()
    self.optimizer = optim.SGD(self.model.parameters(), lr=params['lr'])
    self.iteraions = params['iters']

  def learn(self, x, y):
    self.log_step *= self.iteraions
    for t in range(self.iteraions):
      # Forward pass: Compute predicted y by passing x to the model
      y_pred = self.model(x)

      # Compute and print loss
      loss = self.criterion(y_pred, y)
      if t % self.log_step == self.log_step - 1:
        logger.info(f'#{t} loss:{loss.item()}')
      else:
        logger.debug(f'#{t} loss:{loss.item()}')

      # Zero gradients, perform a backward pass, and update the weights.
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

  def save(self, path):
    torch.save(self.model.state_dict(), path)

  def load(self, path, params):
    self.model = Net(params)
    self.model.load_state_dict(torch.load(path))
    self.model.eval()
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import datetime
from .model import Model

class Net(nn.Module):
  def __init__(self, layers):
    super(Net, self).__init__()
    self.layers = []
    rng = range(len(layers)-1)
    for i in rng:
      self.layers.append(torch.nn.Linear(layers[i]['w'], layers[i+1]['w']))

  # x represents our data
  def forward(self, x):
    for l in self.layers:
      x = l(x)
    return x

class ModelPyTorch(Model):
  name = "pytorch"

  def __init__(self, params):
    super().__init__()
    self.model = Net(params['layers'])
    self.criterion = nn.MSELoss(reduction='sum')
    self.optimizer = optim.SGD(self.model.parameters(), lr=params['lr'])
    self.iteraions = params['iters']

  def learn(self, x, y):
    for t in range(self.iteraions):
      # Forward pass: Compute predicted y by passing x to the model
      y_pred = self.model(x)

      # Compute and print loss
      loss = self.criterion(y_pred, y)
      if t % 100 == 99:
          print(t, loss.item())

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
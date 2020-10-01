import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .model import Model

class Net(nn.Module):
  def __init__(self, d_in, d_out, h):
    super(Net, self).__init__()
    self.input_linear = torch.nn.Linear(d_in, h)
    self.middle_linear = torch.nn.Linear(h, h)
    self.output_linear = torch.nn.Linear(h, d_out)

  # x represents our data
  def forward(self, x):
    y = self.input_linear(x).clamp(min=0)
    y = self.middle_linear(y)
    y = self.output_linear(y)
    return y

class ModelPyTorch(Model):
  name = "pytorch"

  def __init__(self, **params):
    super().__init__()
    self.model = Net(params['d_in'], params['d_out'], params['d_h'])
    self.criterion = nn.MSELoss(reduction='sum')
    self.optimizer = optim.SGD(self.model.parameters(), lr=params['lr'])
    self.iteraions = params['iters']

  def learn(self):
    for t in range(self.iteraions):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = self.model(x)

        # Compute and print loss
        loss = criterion(y_pred, y)
        if t % 100 == 99:
            print(t, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
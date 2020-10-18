import torch
import torch.nn as nn
import torch.nn.functional as F

class Model():
  """
  Base class for model entity (aka IModel)
  """
  name = "base"

  """
  Learn model to ./models/<name>/models/<datetime>
  """
  def learn(self):
    pass

  """
  Predict from ./models/<name>/input/ to ./models/<name>/output/
  """
  def predict(self):
    pass

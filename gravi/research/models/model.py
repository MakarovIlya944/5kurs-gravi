import torch
import torch.nn as nn
import torch.nn.functional as F

class Model():
  """
  Base class for model entity (aka IModel)
  """
  name = "base"

  """
  Learn model
  """
  def learn(self):
    pass

  """
  Predict from ./models/<name>/input/ to ./models/<name>/output/
  """
  def predict(self):
    pass
  
  """
  Save model to ./models/<name>/<datetime>
  """
  def save(self):
    pass
  
  """
  Load model from <path>
  """
  def load(self):
    pass

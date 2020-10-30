from .model import Model
from config import get_logger

logger = get_logger(__name__)

class Spider(Model):
  name = 'spider'
  path = 'gravi/neuralNet/Mnist/bin/Debug/netcoreapp3.0'

  def __init__(self, param):
    super().__init__()

  def learn(self, path):
    print('learn')

    


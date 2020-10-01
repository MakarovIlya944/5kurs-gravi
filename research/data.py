from reverse.builder import *
from reverse.solver import Solver
from scipy import interpolate

class DataCreator():

  name = ''

  def __init__(self, **params):
    super().__init__()
    self.name = params['name']
    net_random_params = params['net']

  def create_pure_data(self, receptors, net_random_params):
    net = center_build(net_random_params)
    s = Solver(receptors=receptors)
    return s.profile(net), net

  def intrepolate_net(self, receptors, dGz):
    # x = [0,1,2];  y = [0,3]; z = [[1,2,3], [4,5,6]]
    
    interpolate.interp2d(x, y, z, kind='cubic')

  def create_data(self, size):
    dGz, net = self.create_pure_data()
    

if __name__ == '__main__':
  params = {
    'name': 'test',
    'net': {

    }
  }
  size = 100
  d = DataCreator(params)
  d.create_data(size)
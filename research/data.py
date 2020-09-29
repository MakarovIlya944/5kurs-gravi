from reverse.builder import *
from reverse.solver import Solver
from scipy import interpolate

class DataCreator():

  name = ''
  net_random_params = {}

  def create_pure_data(self, receptors):
    net = random_build(self.net_random_params)
    s = Solver(receptors=receptors)
    return s.profile(net), net

  def intrepolate_net(self, receptors, dGz):
    # x = [0,1,2];  y = [0,3]; z = [[1,2,3], [4,5,6]]
    
    interpolate.interp2d(x, y, z, kind='cubic')

  def create_data(self, size):
    print()

if __name__ == '__main__':
  d = DataCreator()
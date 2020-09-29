from reverse.builder import *
from reverse.solver import Solver


class DataCreator():

  name = ''
  net_random_params = {}

  def create_pure_data(self, receptors):
    net = random_build(self.net_random_params)
    s = Solver(receptors=receptors)
    return s.profile(net), net

  def intrepolate_net(self, receptors, dGz):
    pass

  def create_data(self, size):
    print()

if __name__ == '__main__':
  d = DataCreator()
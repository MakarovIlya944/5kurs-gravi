from forward import Forward
from net import Net

class Minimizator():

  solver = ''

  alpha = 0
  gamma = []
  n = ''

  def __init__(self):
    self.solver = Forward()
    self.solver.build()
    self.n = Net()

    self.alpha = 0
    

  def minimization(self, maxSteps=1000, eps=1E-10):
    i = 0
    F = functional()
    while i < maxSteps and F < eps:
      F = functional()
      i += 1

  def functional(self):
    self.solver.calculate()
    p = self.result()
    y = [i*i for i in p].sum() * self.alpha
    
    return 1

  def result(self):
    return []
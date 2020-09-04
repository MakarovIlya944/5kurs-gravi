from forward import Forward
from builder import Build

class Minimizator():

  solver = ''

  alpha = 0
  gamma = ''
  n = ''

  def __init__(self, **params):
    self.solver = Forward()
    self.solver.build()
    self.n = Build.build(params.get('net'))
    a = params.get('alpha')
    self.alpha = a if a else 0
    self.gamma = Build.build(params.get('gamma'))

  def minimization(self, maxSteps=1000, eps=1E-10):
    i = 0
    F = functional()
    # TODO add logger
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
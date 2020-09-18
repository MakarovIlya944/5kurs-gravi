from solver import Solver
from builder import Build
from numpy.linalg import norm
from numpy.polynomial.legendre import leggauss
from log import Logger

class Minimizator():

  solver = Solver()
  logger = Logger(10,1)

  alpha = 0
  gamma = ''
  net = ''
  correct = ''

  def __init__(self, **params):
    gamma = params.get('gamma')
    gamma = gamma if gamma else {}
    self.net = Build.build(params.get('net'))
    self.correct = Build.build(params.get('correct'))
    self.gamma = Build.build(params=gamma)
    a = params.get('alpha')[0]
    self.alpha = a if a else 0

    self.solver = Solver(params.get('receptors'), self.correct, self.alpha, self.gamma)

  def minimization(self, maxSteps=1, eps=1E-10):
    calc = []
    for i, p in self.net:
      calc.append(p)
    e = self.error(calc)
    i = 0
    self.logger.n = maxSteps
    while i < maxSteps and e > eps:
      calc = self.solver.solve(self.net)
      e = self.error(calc)
      self.logger.log(str(e))
      i += 1
    return self.net

  def error(self, calc):
    er = []
    for i, (I, p) in enumerate(self.correct):
      er.append(calc[i] - p)
    return norm(er)

  def removeZero(self):
    for I, p in self.net:
      if p < 0:
        self.net[I] = 0
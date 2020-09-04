from forward import Forward
from builder import Build
from numpy.linalg import solve
from numpy.polynomial.legendre import leggauss

class Minimizator():

  solver = ''

  alpha = 0
  gamma = ''
  n = ''

  gauss = 6

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
    Dg = self.result()

    x = 0
    y = sum([i*i for i in P]) * self.alpha
    z = 0

    for p in P:
      _z = 0
      for a in P.around(P.i):
        _z += (P[a] - P[p]) * (P[a] - P[p])
      _z *= self.gamma[p]
      z += _z

    return 1

  def result(self):
    return Build.build()

  def g(self, i):
    x, w = leggauss(self.gauss)
    # Translate x values from the interval [-1, 1] to [a, b]


    t = 0.5*(x + 1)*(b - a) + a
    gauss = sum(w * f(t)) * 0.5*(b - a)
    return 1
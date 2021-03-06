from .net import Net
from numpy import pi, array, prod, reshape, save
from numpy.linalg import norm, solve
from numpy.polynomial.legendre import leggauss
import copy
from config import get_logger, log_config

logger = get_logger(__name__)

class Solver():
  receptors = []
  dGz = []
  correct = Net()
  K = -1
  mesh = -1
  A = array([[], []])
  b = array([])

  alpha = 0
  gamma = {}
  isGamma  = False

  gauss = 6

  def __init__(self, receptors=[], net=Net(), a=0, g=None):
    if len(receptors) == 0:
      return
    self.alpha = a
    if g:
      self.gamma = g
      self.isGamma = True
    self.receptors = receptors
    self.correct = net
    self.K = prod(net.n)
    mesh = 1
    for i in net.d:
      mesh *= i
    mesh /= 4 * pi
    self.mesh = mesh
    logger.debug('calculating receptors dGz')
    L = len(receptors)
    l = int(L * log_config['solver']['dgz'])
    self.dGz = []
    for i, r in enumerate(receptors):
      self.dGz.append(0)
      if not i % l:
        logger.debug(f'#{i}/{L} dGz {i/float(L)*100:.1f}%')
      for j, p in net:
        self.dGz[i] += self._dGz(r, j) * p

  def g(self, i):
    x, w = leggauss(self.gauss)
    # Translate x values from the interval [-1, 1] to [a, b]

    t = 0.5*(x + 1)*(b - a) + a
    gauss = sum(w * f(t)) * 0.5*(b - a)
    return 1

  # if net=None dg with correct net
  # i - source, k - cell
  def _dGz(self, i, k, net=None):
    net = net if net else self.correct

    r = array([k[i] * net.d[i] + net.d[i] / 2 + net.c[i] for i in range(3)])
    r = r - i
    n = norm(r)
    t = self.mesh / (n**3)
    return t * r[2] # only z

  def profile(self, net=None):
    if not net:
      net = self.correct
    res = []
    for r in self.receptors:
      tmp = []
      for i, p in net:
        tmp.append(self._dGz(r, i, net) * p)
      res.append(sum(tmp))
    return res

  def solve(self, net=Net()):
    K = prod(net.n)
    if K != self.K:
      raise Exception("K not correct!")
    A = []
    B = []
    jnet = copy.deepcopy(net)
    l = int(K * K * log_config['solver']['solve'])
    ll = 0
    logger.debug(f'A: {K}x{K} b: {K}')
    for i, pi in net:
      for j, pj in jnet:
        a = sum([self._dGz(s,i,net) * self._dGz(s,j,jnet) for s in self.receptors])
        if i == j:
          a += self.alpha
          if self.isGamma:
            # получение соседних ячеек с i-ой
            around = net.around(i)
            a += len(around) * self.gamma[i] + sum([self.gamma[r] for r in around])
        elif self.isGamma:
          a -= (self.gamma[i]+self.gamma[j])
        A.append(a)
        if ll % l == l - 1:
          logger.debug(f'Combine matrix A {ll/float(K*K)*100:.1f}%')
        ll += 1
      B.append(sum([ self._dGz(s,i,net)*self.dGz[k] for k,s in enumerate(self.receptors)]))
    A = array(A)
    A = A.reshape(int(K),int(K))
    B = array(B)
    logger.debug(f'Begin solving SLAE')
    try:
      res = solve(A,B)
    except Exception as ex:
      with open('matrix.npy', 'wb') as f:
        save(f, A)
      with open('right.npy', 'wb') as f:
        save(f, B)
      raise ex
    for i, (I, p) in enumerate(net):
      net[I] = res[i]
    return res

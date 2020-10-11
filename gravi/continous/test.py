import unittest
from .spline import spline
import numpy as np

def setUp():
  X = np.arange(-1000, 2000, 500)
  Y = np.arange(1000, 5000, 1000)
  X, Y = np.meshgrid(X, Y)
  Z = X + Y
  np.savetxt('test.txt', Z)

X = np.arange(0, 1000, 200)
Y = np.arange(0, 1000, 200)
X, Y = np.meshgrid(X, Y)
Z = X + Y

class TestSpline(unittest.TestCase):

  def test_interpolate(self):
    s = spline('test.txt', [1,1])
    s.Calculate()
    global X, Y, Z
    x, y, z = s.Interpolate(X, Y)
    self.assertEquals(x, X)

def test():
  setUp()
  unittest.main()

if __name__ == '__main__':
  test()
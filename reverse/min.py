
class Minimizator():

  def Minimization(self, maxSteps=1000, eps=1E-10):
    i = 0
    F = Functional()
    while i < maxSteps and F < eps:
      F = Functional()
      i += 1

  def Functional(self):
    return 1
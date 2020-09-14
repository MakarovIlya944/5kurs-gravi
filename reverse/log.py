
class Logger(object):
  i = 0
  n = 1000
  step = 10

  def __init__(self, n=1000, step=10):
    self.n=n
    self.step=step

  def log(self, msg):
    print(f'{self.i}/{self.n}: ' + msg)
    self.i += self.step
    if self.i >= self.n:
      self.i = 0
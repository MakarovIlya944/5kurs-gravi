from net import Net

class Build():

  def build(values=None):
    n = Net()
    if values:
      for k in values:
        n[k] = values[k]
    return n
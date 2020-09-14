from net import Net

class Build():

  def simple(values=None):
    n = Net()
    if values:
      for k in values:
        n[k] = values[k]
    return n

  def build(**params):
    values = params.get('values')
    
    count = params.get('count')
    count = count if count else (10,10,10)

    border = params.get('border')
    border = border if border else (1000,1000,1000)

    v = params.get('v')
    v = v if v else 0

    n = Net(count=count,border=border,v=v)
    if values:
      for k in values:
        n[k] = values[k]
    return n
from net import Net

def simple_build(values=None):
  n = Net()
  if values:
    for k in values:
      n[k] = values[k]
  return n

def complex_build(**params):
  values = params['params'].get('values')
  
  count = params['params'].get('count')
  count = count if count else (10,10,10)

  border = params['params'].get('border')
  border = border if border else (1000,1000,1000)

  v = params['params'].get("default")
  v = v if v else 0

  center = params['params'].get('center')
  center = center if center else (0,0,0)

  n = Net(count=count,border=border,center=center,v=v)
  if values:
    for k in values:
      n[k] = values[k]
  return n

def random_build(**params):
  pass
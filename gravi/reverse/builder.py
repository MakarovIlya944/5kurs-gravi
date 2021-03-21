from numpy.linalg.linalg import norm
from .net import Net
from numpy import array

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

  border = params['params'].get('right')
  border = border if border else (1000,1000,1000)

  v = params['params'].get("default")
  v = v if v else 0

  center = params['params'].get('left')
  center = center if center else (0,0,0)

  n = Net(count=count,border=border,center=center,v=v)
  if values:
    for k in values:
      n[k] = values[k]
  return n

def center_build(params):
  count = params.get('count')
  count = count if count else (10,10,10)

  width = array(params.get('width'))
  center = array(params.get('center'))
  c_value = params.get('c_value')

  inds = center - width
  inds_l = []
  for ind,i in enumerate(inds):
    if i < 0:
      inds_l.append(0)
      continue
    if i >= count[ind]:
      inds_l.append(count[ind]-1)
      continue
    inds_l.append(i)

  inds = center + width
  inds_r = []
  for ind,i in enumerate(inds):
    if i < 0:
      inds_r.append(0)
      continue
    if i >= count[ind]:
      inds_r.append(count[ind]-1)
      continue
    if inds_l[ind] > i:
      inds_r.append(inds_l[ind])
      continue
    inds_r.append(i)

  vals = [
    [i for i in range(inds_l[c_i], inds_r[c_i]+1)]
    for c_i in range(3)
  ]

  values = {}
  for x in vals[0]:
    for y in vals[1]:
      for z in vals[2]:
        values[(x,y,z)] = c_value

  border = params.get('right')
  border = border if border else (1000,1000,1000)

  v = params.get("default")
  v = v if v else 0

  center = params.get('left')
  center = center if center else (0,0,0)

  n = Net(count=count,border=border,center=center,v=v)
  if values:
    for k in values:
      n[k] = values[k]
  return n

def line_build(params):
  count = params.get('count') or (10,10,10)
  line_begin = array(params.get('line_begin'))
  length = array(params.get('length'))
  c_value = params.get('c_value')
  border = params.get('right') or (1000,1000,1000)
  center = params.get('left') or (0,0,0)
  v = params.get("default") or 0

  values = {}
  for x in range(line_begin[0], length + line_begin[0]):
    values[(x,line_begin[1],line_begin[2])] = c_value

  n = Net(count=count,border=border,center=center,v=v)
  if values:
    for k in values:
      n[k] = values[k]
  return n

def circle_build(params):
  count = array(params.get('count') or (10,1,10))
  center = count/2
  radius = params.get('radius') or 2
  if count[1] != 1:
    raise IndexError("Circle build only flat(y=1) nets")
  c_value = params.get('c_value')
  right = params.get('right') or (1000,1000,1000)
  left = params.get('left') or (0,0,0)
  v = params.get("default") or 0

  n = Net(count=count,border=right,center=left,v=v)

  for x in range(count[0]):
    for z in range(count[2]):
      if norm(array([x,0.5,z])-center) <= radius:
        n[(x,0,z)] = c_value

  return n
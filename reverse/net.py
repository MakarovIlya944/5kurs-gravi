from numpy import ndarray
class Net():
  cells = ""
  border = (1000,1000,1000)
  n = (10,10,10)
  i = [0,0,0]
  d = (100,100,100)

  def __init__(self, count=(10,10,10), border=(1000,1000,1000), v=0):
    self.n = count
    self.d = tuple([border[i]/count[i] for i in range(3)])
    self.cells = ndarray(count)
    for i in range(count[0]):
      for j in range(count[1]):
        for k in range(count[2]):
          self.cells[i][j][k] = v

  def save(self):
    lines = []
    lines.append(f'{self.border[0]} {self.border[1]} {self.border[2]}')
    lines.append(f'{self.n[0]} {self.n[1]} {self.n[2]}')
    for i in range(self.n[0]):
      for j in range(self.n[1]):
        for k in range(self.n[2]):
          lines.append(str(self.cells[i][j][k]))
    
    with open('file.txt', 'w') as f:
      f.writelines(lines)

  def around(self, key):
    r = []
    for i in range(-1,1):
      for j in range(-1,1):
        for k in range(-1,1):
          if not (i == j and j == k):
            r.append((key[0] + i, key[1] + j, key[2] + k))
    r = [el for el in r if self.__correctindex(el)]
    return r

  def __correctindex(self, key):
    for i,v in enumerate(key):
      if v < 0 or v > self.n[i]:
        return False
    return True

  def __getitem__(self, key):
    return self.cells[key[0]][key[1]][key[2]]

  def __setitem__(self, key, value):
    self.cells[key[0],key[1],key[2]] = value

  def __iter__(self):
    self.i = [-1,0,0]
    return self

  def __next__(self):
    if self.i[0] < self.n[0]-1:
      self.i[0] += 1
    else:
      if self.i[1] < self.n[1]-1:
        self.i[1] += 1
      else:
        if self.i[2] < self.n[2]-1:
          self.i[2] += 1
        else:
          self.i[2] = 0
          raise StopIteration
        self.i[1] = 0
      self.i[0] = 0

    return (self.i, self.cells[self.i[0]][self.i[1]][self.i[2]])

  def __str__(self):
    s = ''
    for i, m in enumerate(self.cells):
      s += 'zi:' + str(i) + '\n'
      s += str(m) + '\n'
    return s

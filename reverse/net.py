
class Net():
  cells = ""
  border = (1000,1000,1000)
  n = (10,10,10)

  def __init__(self, x=10, y=10, z=10, v=0):
    self.n = (x,y,z)
    for i in range(x):
      for j in range(y):
        for k in range(z):
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
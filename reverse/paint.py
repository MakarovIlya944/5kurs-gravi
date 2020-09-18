import matplotlib.pyplot as plt

def matrix(m):
  a = m.shape
  m = m.reshape(a[0],a[2])
  plt.matshow(m,vmax=10,vmin=0)
  plt.show()

def profile(x, y):
  plt.plot(x,y)
  plt.show()
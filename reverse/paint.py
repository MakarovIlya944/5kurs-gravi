import matplotlib.pyplot as plt

def matrix(m, max=10):
  a = m.shape
  m = m.reshape(a[0],a[2])
  m = m.transpose()
  plt.matshow(m,vmax=max,vmin=0)
  plt.show()

def profile(x, y):
  plt.plot(x,y)
  plt.show()
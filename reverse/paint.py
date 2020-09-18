import matplotlib.pyplot as plt

class Painter():

  def matrix(m):
    a = m.shape
    m = m.reshape(a[0],a[2])
    plt.matshow(m)
    plt.show()
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot
from .data import DataReader
from mpl_toolkits.mplot3d import axes3d, Axes3D
from .models.pytorch import ModelPyTorch

def show_nets(name, params):
  x,y,c = DataReader.read_folder('data/' + name)

  fig, axs = plt.subplots(3, 3, sharex=True, sharey=True)
  for i in range(9):
    if c[i][1] == 1:
      _x = c[i][0]
      _y = c[i][2]
      m = [y[i][k*_x:(k+1)*_x] for k in range(_y)]
      j = i // 3
      k = i % 3
      axs[j, k].matshow(m,vmax=10,vmin=0)
      axs[j, k].set_title(str(i))
  plt.tight_layout()
  plt.show()

def show_3d(name):
  x,y,c = DataReader.read_folder('data/' + name)

  dx, dy, dz = 5, 5, 5
  for i in range(len(c)):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    m = max(y[i])
    for j, v in enumerate(y[i]):
      ax.scatter(j % c[i][0] * dx, j % (c[i][0] * c[i][1]) // c[i][1] * dy, j // (c[i][0] * c[i][1])  * -dz, marker='o', c=[[0.1,0.1,v/m]])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

def show_loss(name):
  x = []
  y = []
  for k in range(1, 5):
    with open(name + str(k) + '.log', 'r') as f:
      ll = f.readlines()
      _y = []
      _x = []
      j = 0
      for l in ll:
        i = l.index('loss:')
        _y.append(float(l[i+5:]))
        _x.append(j)
        j+=1
    x.append(_x)
    y.append(_y)
  plot(x[0],y[0],x[1],y[1],x[2],y[2],x[3],y[3])
  plt.show()

def test_nn():
  mp = ModelPyTorch()
  print(mp.model)
  print(mp.model.parameters())
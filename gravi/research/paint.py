import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d, Axes3D
import traceback
import numpy.linalg as l
from .data import DataReader

def show_predict(predicted_data, model_index=None, dataset_index=None, paint_show=None, paint_3d=None):
  data = predicted_data['data']
  if model_index and dataset_index and paint_show:
    try:
      x,y,c = DataReader.read_folder('data/' + data[model_index]['dataset'])
      true_y = y[dataset_index]
      pred_y = data[model_index]['predicted'][dataset_index]
      c = c[dataset_index]
      vm = max(true_y)

      if paint_3d:
        temp, points, true, calc, pred = [], [], [], [], []
        for i in range(c[2]):
          true_z = true_y[i*c[0]*c[1]:(i+1)*c[0]*c[1]]
          pred_z = pred_y[i*c[0]*c[1]:(i+1)*c[0]*c[1]]
          cacl_z = [abs(true_z[l] - pred_z[l]) for l in range(len(true_z))]
          calc.append([cacl_z[k*c[0]:(k+1)*c[0]] for k in range(c[1]-1)])
          true.append([true_z[k*c[0]:(k+1)*c[0]] for k in range(c[1]-1)])
          pred.append([pred_z[k*c[0]:(k+1)*c[0]] for k in range(c[1]-1)])
        if paint_show == 'true':
          temp = true
        elif paint_show == 'pred':
          temp = pred
        elif paint_show == 'calc':
          temp = calc
        else:
          raise Exception("Wrong type of show")
        for i,ppp in enumerate(temp):
          for j,pp in enumerate(ppp):
            for k,p in enumerate(pp):
              if p > 1e-2:
                points.append((k, j, i, int((p/vm)*255)))
        x, y, z, filled_2, fcolors_2 = strange_magic(points, c)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.voxels(x, y, z, filled_2, facecolors=fcolors_2)
      else:
        n_plotsx, n_plotsy = 3, 4
        fig, axs = plt.subplots(n_plotsx, n_plotsy)
        for i in range(c[2]):
          true_z = true_y[i*c[0]*c[1]:(i+1)*c[0]*c[1]]
          pred_z = pred_y[i*c[0]*c[1]:(i+1)*c[0]*c[1]]
          cacl_z = [abs(true_z[l] - pred_z[l]) for l in range(len(true_z))]

          j = i // 4
          k = i % 4
          m = [cacl_z[k*c[0]:(k+1)*c[0]] for k in range(c[1]-1)]
          if paint_show == 'true':
            m = [true_z[k*c[0]:(k+1)*c[0]] for k in range(c[1]-1)]
          elif paint_show == 'pred':
            m = [pred_z[k*c[0]:(k+1)*c[0]] for k in range(c[1]-1)]
          else:
            raise Exception("Wrong type of show")
          axs[j, k].matshow(m,vmax=vm,vmin=0,cmap="Reds")
          axs[j, k].set_title(f'z: {-i}')
        plt.tight_layout()
    except IndexError as e:
      traceback.print_exc()
      raise e
  else:
    x = [i for i in range(len(data))]
    y = [d['l2_diff'] for d in data]
    labels = [s['name'] for s in data]
    plt.bar(x,y,tick_label=labels)
  plt.show()

def strange_magic(points, c):
  # build up the numpy logo
  n_voxels = np.zeros(c, dtype=bool)
  basecolor = '#FF0000'
  facecolors = np.where(n_voxels, basecolor + 'FF', '#00000000')
  for p in points:
    alpha = str(hex(p[3]))[2:]
    if p[3] < 16:
      alpha = '0' + alpha
    facecolors[p[0]][p[1]][p[2]] = basecolor + alpha
  filled = np.ones(n_voxels.shape)

  # upscale the above voxel image, leaving gaps
  filled_2 = explode(filled)
  fcolors_2 = explode(facecolors)

  # Shrink the gaps
  x, y, z = np.indices(np.array(filled_2.shape) + 1).astype(float) // 2
  x[0::2, :, :] += 0.05
  y[:, 0::2, :] += 0.05
  z[:, :, 0::2] += 0.05
  x[1::2, :, :] += 0.95
  y[:, 1::2, :] += 0.95
  z[:, :, 1::2] += 0.95

  return x, y, z, filled_2, fcolors_2

def explode(data):
    size = np.array(data.shape)*2
    data_e = np.zeros(size - 1, dtype=data.dtype)
    data_e[::2, ::2, ::2] = data
    return data_e

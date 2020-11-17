import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d, Axes3D
import traceback
import numpy.linalg as l
from .data import DataReader

def show_predict(predicted_data, model_index=None, dataset_index=None, paint_show=None, paint_3d=None):
  data = predicted_data['data']
  if not (model_index is None or dataset_index is None or paint_show is None):
    try:
      x,y,c = DataReader.read_folder('data/' + data[model_index]['dataset'])
      true_y = y[dataset_index]
      pred_y = data[model_index]['predicted'][dataset_index]
      c = c[dataset_index]
      vm = max(true_y)

      points, true, calc, pred = [], [], [], []
      for i in range(c[2]):
        true_z = true_y[i*c[0]*c[1]:(i+1)*c[0]*c[1]]
        pred_z = pred_y[i*c[0]*c[1]:(i+1)*c[0]*c[1]]
        cacl_z = [abs(true_z[l] - pred_z[l]) for l in range(len(true_z))]
        calc.append([cacl_z[k*c[0]:(k+1)*c[0]] for k in range(c[1]-1)])
        true.append([true_z[k*c[0]:(k+1)*c[0]] for k in range(c[1]-1)])
        pred.append([pred_z[k*c[0]:(k+1)*c[0]] for k in range(c[1]-1)])
      if not paint_3d is None:
        if paint_show == 'mix':
          pred = normalize(pred)
          true = normalize(true)
          x, y, z, filled_2, fcolors_2 = strange_magic(pred.astype(np.int32), true.astype(np.int32))
        else:
          if paint_show == 'true':
            points = true
          elif paint_show == 'pred':
            points = pred
          elif paint_show == 'calc':
            points = calc
          else:
            raise Exception(f"Wrong {paint_show} type of show")
          points = normalize(points)
          x, y, z, filled_2, fcolors_2 = strange_magic(points.astype(np.int32))

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.voxels(x, y, z, filled_2, facecolors=fcolors_2)
      else:
        n_plotsx, n_plotsy = 3, 4
        fig, axs = plt.subplots(n_plotsx, n_plotsy)
        for i in range(c[2]):
          j = i // n_plotsy
          k = i % n_plotsy
          if paint_show == 'calc':
            m = calc[i]
          elif paint_show == 'true':
            m = true[i]
          elif paint_show == 'pred':
            m = pred[i]
          else:
            raise Exception(f"Wrong {paint_show} type of show")
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

def normalize(points, eps=1e-2):
  points = np.array(points)
  vm = np.max(points)
  for i,p in np.ndenumerate(points):
    if p > eps:
      points[i] = int((p/vm)*255)
    else:
      points[i] = int(0)
  return points

def strange_magic(pred_p, true_p=None):
  facecolors = pred_p.astype(str)
  # np.unicode_, 16
  basecolor = '#FF0000'
  for i, p in np.ndenumerate(pred_p):
    alpha = str(hex(p))[2:]
    if p < 16:
      alpha = '0' + alpha
    facecolors[i] = basecolor + alpha
  if not true_p is None:
    basecolor = '#0000FF'
    for i, p in np.ndenumerate(true_p):
      if p > 16:
        alpha = str(hex(p))[2:]
        facecolors[i] = basecolor + alpha
  filled = np.ones(facecolors.shape)

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

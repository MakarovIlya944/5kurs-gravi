import matplotlib.pyplot as plt
import numpy.linalg as l
from .data import DataReader


def show_predict(predicted_data, model_index=None, dataset_index=None, paint_show=None):
  data = predicted_data['data']
  if model_index and dataset_index and paint_show:
    try:
      x,y,c = DataReader.read_folder('data/' + data[model_index]['dataset'])
      true_y = y[dataset_index]
      pred_y = data[model_index]['predicted'][dataset_index]
      c = c[dataset_index]
      vm = max(true_y)

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
        axs[j, k].matshow(m,vmax=vm,vmin=0,cmap="Reds")
        axs[j, k].set_title(f'z: {-i}')
      plt.tight_layout()
    except IndexError as e:
      print(str(e))
      raise e
  else:
    x = [i for i in range(len(data))]
    y = [d['l2_diff'] for d in data]
    labels = [s['name'] for s in data]
    plt.bar(x,y,tick_label=labels)
  plt.show()

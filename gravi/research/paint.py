import matplotlib.pyplot as plt
import numpy.linalg as l
from .data import DataReader


def show_predict(predicted_data, model_index=None, dataset_index=None):
  data = predicted_data['data']
  x = [i for i in range(len(data))]
  y = [d['l2_diff'] for d in data]
  labels = [s['name'] for s in data]
  if model_index and dataset_index:
    x,y,c = DataReader.read_folder('data/' + data[model_index]['dataset'])
    true_y = y[dataset_index]
    pred_y = data[model_index]['predicted'][dataset_index]
    c = c[dataset_index]
    dx, dy, dz, sm = 10, 10, 10, 1
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    m = max(true_y)
    for j, v in enumerate(true_y):
      ax.scatter(j % c[0] * dx, j % (c[0] * c[1]) // c[1] * dy, j // (c[0] * c[1])  * -dz, marker='o', c=[[1,0,0,abs(pred_y[j] - v)/m]])
      ax.scatter(j % c[0] * dx + sm, j % (c[0] * c[1]) // c[1] * dy + sm, j // (c[0] * c[1])  * -dz + sm, marker='o', c=[[0,0,1,v/m]])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
  else:
    plt.bar(x,y,tick_label=labels)
  plt.show()

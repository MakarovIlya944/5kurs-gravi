from min import Minimizator
from numpy import array,ndarray
from paint import Painter
from copy import copy

# mobilenet, vg16, inseption v3 (CNN)
# 1 profile, RNN
# RvNN
# normalize

def main():
  print('Let\'s go!')

  # сетка задается через несколько параметров
  # count кортеж с кол-вом ячеек по каждой из осей
  # border дальняя граница по каждой из осей. начало в 0,0,0
  # v значение по умолчанию для каждой ячейки
  # values точечные замены значений, задается как индекс ячейки: значение
  defNet = {
    'count':(10,1,6),
    'border':(5000,10000,-1500),
    'center': (0,0,0)
  }
  # начальная сетка
  net = copy(defNet)
  net['v'] = 0.1
  # правильная сетка
  correct = copy(defNet)
  correct['values'] = {
      (4,0,2): 1,
      (5,0,2): 1,
      (4,0,3): 1,
      (5,0,3): 1,
    }
  receptors = [array([i*12,5000,0]) for i in range(100)]
  alpha=1,
  gamma=None
  # параметры гамма задаются как сетка со значениями для каждой ячейки
  # gamma={
  #   'count':(2,2,2),
  #   'border':(8,8,8),
  #   'values': {
  #     (1,0,0): 1,
  #     (1,0,1): 1,
  #     (1,1,0): 1,
  #     (1,1,1): 1,
  #   }
  # }

  smile = Minimizator(net=net, receptors=receptors, correct=correct, alpha=alpha, gamma=gamma)
  net = smile.minimization()

  print(str(net))
  Painter.matrix(net.cells)

  print('Good bye!')

if __name__ == '__main__':
  main()
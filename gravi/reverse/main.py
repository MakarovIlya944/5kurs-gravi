from .min import Minimizator
from .paint import *
from numpy import array,ndarray
from copy import copy
import matplotlib.pyplot as plt

# mobilenet, vg16, inseption v3 (CNN)
# resNet
# LSTM - experemental 
# сетки для NLP 
# использовать связь рядом стоящих рецепторов
# возвращает в глубину
# двунаправленая!

# повышение качества
# для предсказания
# разные данные приводишь по разному к входными
# прогоняешь
# берешь среднее

# посчитать для каждой ячейки уверенность
# и на тех прогонах где было больше истинности
# добавить в датасет и дообучить
# те которые были правильные так и поставить
# а те которые ошиблись ставить другие

# делать пока есть изменения т.е. пока есть уверенные точки

def main():
  print('Let\'s go!')

  # сетка задается через несколько параметров
  # count кортеж с кол-вом ячеек по каждой из осей
  # border дальняя граница по каждой из осей. начало в 0,0,0
  # v значение по умолчанию для каждой ячейки
  # values точечные замены значений, задается как индекс ячейки: значение
  defNet = {
    'count': (5,1,5),
    'right': (3000,50,-1500),
    'left': (1000,0,-500)
  }
  # начальная сетка
  net = copy(defNet)
  # net["default"] = 0.1
  # правильная сетка
  correct = copy(defNet)
  correct['values'] = {
      (1,0,3): 10,
      (2,0,3): 10,
      (3,0,3): 10,
    }
  receptors = [array([i*50-1000,0,0]) for i in range(100)]
  alpha=2,
  gamma=None
  # параметры гамма задаются как сетка со значениями для каждой ячейки
  gamma = copy(defNet)
  gamma={
    'values': {
      (1,0,3): 1,
      (2,0,3): 1,
      (3,0,3): 1,

      (2,0,4): 0.1,
      (2,0,2): 0.1,
      (0,0,3): 0.01,
      (4,0,3): 0.01,

      # (1,0,2): 0.1,
      # (4,0,2): 0.1,
      # (1,0,3): 0.1,
      # (4,0,3): 0.1,
    }
  }

  smile = Minimizator(net=net, receptors=receptors, correct=correct, alpha=alpha, gamma=gamma)
  net = smile.minimization()
  prof = smile.solver.profile
  dG = prof(net)
  dGCorrect = prof()

  print(str(net))
  # profile([r[0] for r in receptors], dG)
  m = 0
  c = correct['values']
  for k in c:
    if m < c.get(k):
      m = c.get(k)
  matrix(net.cells, m)

  plt.plot([r[0] for r in receptors], dG, 'r', [r[0] for r in receptors], dGCorrect)
  plt.show()

  print('Good bye!')

if __name__ == '__main__':
  main()
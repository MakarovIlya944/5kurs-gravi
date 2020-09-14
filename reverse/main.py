from min import Minimizator
from numpy import array

def main():
  print('Let\'s go!')

  # сетка задается через несколько параметров
  # count кортеж с кол-вом ячеек по каждой из осей
  # border дальняя граница по каждой из осей. начало в 0,0,0
  # v значение по умолчанию для каждой ячейки
  # values точечные замены значений, задается как индекс ячейки: значение
  # начальная сетка
  net={
    'count':(2,2,2),
    'border':(8,8,8)
  }
  # правильная сетка
  correct={
    'count':(2,2,2),
    'border':(8,8,8),
    'values': {
      (0,0,0): 1,
      (0,0,1): 1,
      (0,1,0): 1,
      (0,1,1): 1,

      (1,0,0): 1,
      (1,0,1): 1,
      (1,1,0): 1,
      (1,1,1): 1,
    }
  }
  receptors=[
    array([2,2,0]),
    array([4,2,0]),
    array([6,2,0]),

    array([2,4,0]),
    array([4,4,0]),
    array([6,4,0]),

    array([2,6,0]),
    array([4,6,0]),
    array([6,6,0]),
  ]
  alpha=1,
  gamma=None
  # параметры гамма задаются как сетка со значениями для каждой ячейки
  gamma={
    'count':(2,2,2),
    'border':(8,8,8),
    'values': {
      (1,0,0): 1,
      (1,0,1): 1,
      (1,1,0): 1,
      (1,1,1): 1,
    }
  }
  
  smile = Minimizator(net=net, receptors=receptors, correct=correct, alpha=alpha, gamma=gamma)
  net = smile.minimization()

  print(str(net))

  print('Good bye!')

if __name__ == '__main__':
  main()
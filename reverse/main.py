from min import Minimizator
from numpy import array

def main():
  print('Let\'s go!')

  net={
    'count':(2,2,2),
    'border':(8,8,8)
  }  
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
from gravi.research.main import learn,prepare_data
from gravi.research.test import *
import sys
from config import get_logger

logger = get_logger('main')

def test(name):
  # show_3d(name)
  show_loss(name)

def main():
  tmp = ' '.join(sys.argv)
  logger.info('Start: ' + tmp)
  try:
    if sys.argv[1] == 'data':
      config_name = sys.argv[2]
      if len(sys.argv) > 4:
        config_name = sys.argv[4]
      prepare_data(int(sys.argv[3]), sys.argv[2], config_name)
    elif sys.argv[1] == 'learn':
      with open(f'data/{sys.argv[2]}/0_in', 'r') as f:
        i = len(f.readlines())
      with open(f'data/{sys.argv[2]}/0_out', 'r') as f:
        o = len(f.readlines())
      config_name = sys.argv[2]
      if len(sys.argv) > 3:
        config_name = sys.argv[3]
      learn(i, o, sys.argv[2], config_name)
    elif sys.argv[1] == 'test':
      test(sys.argv[2])
  except IndexError:
    print('Invalid args number')
    print('data <dataset name> <dataset size>\tWill be save dataset to ./data/<dataset name>')
    print('learn <dataset name> [<model config name>]\t\t\tWill be save model to ./models/<model type>/<date>')
    print('predict <dataset name>\t\t\tWill be save model to ./models/<model type>/<date>')

if __name__ == '__main__':
  main()
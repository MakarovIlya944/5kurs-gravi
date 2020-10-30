from gravi.research.main import learn,prepare_data
from gravi.research.test import show_3d
import sys
from config import get_logger

logger = get_logger('main')

def test(name):
  show_3d(name)

def main():
  logger.info("Start")
  try:
    if sys.argv[1] == 'data':
      prepare_data(int(sys.argv[3]), sys.argv[2])
    elif sys.argv[1] == 'learn':
      with open(f'data/{sys.argv[2]}/0_in', 'r') as f:
        i = len(f.readlines())
      with open(f'data/{sys.argv[2]}/0_out', 'r') as f:
        o = len(f.readlines())
      learn(i, o, sys.argv[2])
    elif sys.argv[1] == 'test':
      test(sys.argv[2])
  except IndexError as ex:
    print('Invalid args number')
    print('data <dataset name> <dataset size>\tWill be save dataset to ./data/<dataset name>')
    print('learn <dataset name>\tWill be save model to ./models/<model type>/<date>')
    print('predict <dataset name>\tWill be save model to ./models/<model type>/<date>')

if __name__ == '__main__':
  main()
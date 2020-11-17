from gravi.research.main import *
from gravi.research.paint import *
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
    elif sys.argv[1] == 'predict':
      model_index = None
      dataset_index = None
      show_type = None
      show_3d = None
      if len(sys.argv) > 4:
        model_index = int(sys.argv[3])
        dataset_index = int(sys.argv[4])
        if len(sys.argv) > 5:
          show_type = sys.argv[5]
        else:
          show_type = 'calc'
        if len(sys.argv) > 6:
          show_3d = sys.argv[6]
      predicted_data = predict(sys.argv[2], is_save=False)
      show_predict(predicted_data, model_index, dataset_index, show_type, show_3d)
    elif sys.argv[1] == 'test':
      test(sys.argv[2])
  except IndexError:
    print('Invalid args number')
    print('data <dataset name> <dataset size> [<net_config>]\t\tWill be save dataset to ./data/<dataset name>')
    print('learn <dataset name> [<model config name>]\t\tWill be save model to ./models/<model type>/<date>')
    print('predict <predict config name> [<model index in predict config> <net index in dataset> [<abs or true or pred type of paint>]]\t\tPredict data from ./models/<dataset>, models from ./models/<model type>/<model>. Names for dataset and model get from configs/predict/<config name>')

if __name__ == '__main__':
  main()
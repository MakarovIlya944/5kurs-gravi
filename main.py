from gravi.research.main import *
from gravi.research.paint import *
from gravi.research.test import *
import sys
from config import *

logger = get_logger('main')
parsers = get_args_parser()

def test(name):
  # show_3d(name)
  show_loss(name)

def main():
  logger.info('Start: ' + ' '.join(sys.argv[1:]))
  command = sys.argv[1]
  if command == 'data':
    args = vars(parsers["data"].parse_args())
    config_name = args['config']
    dataset_name = config_name
    if args.get('name'):
      dataset_name = args.get('name')
    prepare_data(args['n'], dataset_name, config_name)
  elif command == 'learn':
    args = vars(parsers["learn"].parse_args())
    dataset_name = args['dataset']
    with open(f'data/{dataset_name}/0_in', 'r') as f:
      i = len(f.readlines())
    with open(f'data/{dataset_name}/0_out', 'r') as f:
      o = len(f.readlines())
    config_name = args['config']
    learn(i, o, dataset_name, config_name)
  elif command == 'predict':
    args = vars(parsers["predict"].parse_args())
    save_image = args['save']
    predict_config = args['config']

    model_index = args.get('M')
    dataset_index = args.get('N')
    if (model_index and not dataset_index) or (model_index and not dataset_index):
      logger.error('Need to set dataset and model index both')
      exit(1)
    read_all = True
    if not model_index is None:
      read_all = False
    show_type = args.get('S')
    show_3d = args.get('dim')
    if read_all:
      predicted_data, X, Y, C = predict(predict_config, is_save=False)
    else:
      predicted_data, X, Y, C = predict(predict_config, is_save=False, net_index=dataset_index, model_index=model_index)
    show_predict(predicted_data, model_index, dataset_index, show_type, show_3d, X, Y, C, is_save=save_image)
  elif command == 'test':
    test(sys.argv[2])
  elif command == 'inspect':
    save_image = False
    for s in sys.argv:
      if s == 'save':
        save_image = True
    try:
      sys.argv.remove('save')
    except Exception:
      pass
    Y = calc_stat(sys.argv[2],sys.argv[3])
    paint_solidity(Y, is_save=save_image)
  else:
    if command != '-h' or command != '--help':
      logger.error('Invalid command ' + command)
    parsers['base'].print_help()
      
if __name__ == '__main__':
  main()
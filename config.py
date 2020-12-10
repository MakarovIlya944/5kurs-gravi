import logging
import argparse

_log_format = f"%(asctime)s|%(levelname)s|%(name)s/%(filename)s/%(funcName)s|%(message)s"

def get_file_handler(name):
  file_handler = logging.FileHandler(name)
  file_handler.setLevel(logging.DEBUG)
  file_handler.setFormatter(logging.Formatter(_log_format))
  file_handler.name = 'file_log'
  return file_handler

def get_stream_handler():
  stream_handler = logging.StreamHandler()
  stream_handler.setLevel(logging.DEBUG)
  stream_handler.setFormatter(logging.Formatter(_log_format))
  stream_handler.name = 'stream_log'
  return stream_handler

def get_logger(name, filename='x.log'):
  logger = logging.getLogger(name)
  logger.setLevel(logging.DEBUG)
  logger.addHandler(get_file_handler(filename))
  logger.addHandler(get_stream_handler())
  return logger

def get_args_parser():
  base_parser = argparse.ArgumentParser(description='Research gravi survey solving by diffrent neural nets and compare with reverse calculation')
  subparsers = base_parser.add_subparsers(help='available commands')

  data_parser = subparsers.add_parser('data', help='Dataset generation. It will save to ./data/<dataset name>')
  data_parser.add_argument('command', help='command')
  data_parser.add_argument('config', help='config name')
  data_parser.add_argument('n', help='dataset size', type=int)
  data_parser.add_argument('--name', dest="name", required=False,
                      help='optional dataset name. By default equal config name')

  learn_parser = subparsers.add_parser('learn', help='Learn dataset. It will save model to ./models/<model type>/<date>')
  learn_parser.add_argument('command', help='command')
  learn_parser.add_argument('config', help='model config name')
  learn_parser.add_argument('dataset', help='dataset name')

  predict_parser = subparsers.add_parser('predict', help='Predict results')
  predict_parser.add_argument('command', help='command')
  predict_parser.add_argument('config', type=str, help='predict config name')
  predict_parser.add_argument('-m', type=int, required=False, help='model index in config')
  predict_parser.add_argument('-n', type=int, required=False, help='net index in dataset')
  predict_parser.add_argument('-s', help='show type', choices=['mix','calc','pred','true'] ,default='calc')
  predict_parser.add_argument('--3d', action='store_true',dest='dim', help='preffer 3d show instead of 2d')
  predict_parser.add_argument('--response', action='store_true',dest='resp', help='response predict')
  predict_parser.add_argument('--save', action='store_true',dest='save',
                      help='save generated image instead of showing')

  inspect_parser = subparsers.add_parser('inspect', help='Inspect dataset and logs')
  inspect_parser.add_argument('command', help='command')
  inspect_parser.add_argument('sub', help='sub command')
  inspect_parser.add_argument('config', help='dataset config name')
  inspect_parser.add_argument('--dataset', dest="dataset", required=False, help='dataset name')
  inspect_parser.add_argument('-n', type=int, required=False, help='net index in dataset')
  inspect_parser.add_argument('--model', required=False, help='model name to calculate predicted response')
  inspect_parser.add_argument('--model-config', required=False, help='model config to calculate predicted response')
  inspect_parser.add_argument('--save', action='store_true',dest='save',
                      help='save generated image instead of showing')

  return {
    "base": base_parser,
    "data":data_parser,
    "learn":learn_parser,
    "predict":predict_parser,
    "inspect":inspect_parser,
  }

log_config = {
    'solver': 0.5,
    'data_creation': 0.1,
    'data_read': 100,
    'pytorch': 0.1
}

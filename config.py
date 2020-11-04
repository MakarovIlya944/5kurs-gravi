import logging

_log_format = f"%(asctime)s|%(levelname)s|%(name)s/%(filename)s/%(funcName)s|%(message)s"

def get_file_handler(name):
  file_handler = logging.FileHandler(name)
  file_handler.setLevel(logging.WARNING)
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

log_config = {
    'solver': 0.5,
    'data_creation': 0.1,
    'pytorch': 0.1
}

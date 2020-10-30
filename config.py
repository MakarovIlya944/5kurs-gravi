import logging

_log_format = f"%(asctime)s|%(levelname)s|%(name)s/%(filename)s/%(funcName)s|%(message)s"

def get_file_handler():
    file_handler = logging.FileHandler("x.log")
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(logging.Formatter(_log_format))
    return file_handler

def get_stream_handler():
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(_log_format))
    return stream_handler

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(get_file_handler())
    logger.addHandler(get_stream_handler())
    return logger

log_config = {
    'solver': 0.2,
    'data_creation': 0.5
}

prepare_data_default_params = {
      'net': {
        'count': (5,5,5),
        'right': (3000,1000,-1500),
        'left': (1000,0,-500),
        'width': {
           'max': (2,2,2),
           'min': (0,0,0)
        },
        'center': {
        #    'max': (1,0,1),
           'min': (0,0,2)
        },
        'c_value': 10,
        },
      'receptors':{
        'x':{
          'r': 3000,
          'l': 1000,
          'n': 20
        },
        'y':{
          'r': 0,
          'l': -2000,
          'n': 20
        }
      }
    }

model_learn_default_params = {
      'iters': 1000,
      'lr':0.01
    }
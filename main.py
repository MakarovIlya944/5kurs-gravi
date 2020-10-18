from gravi.research.data import DataReader, interpolate
import logging
from gravi.research.main import prepare_data

from gravi.research.test import show_nets

logging.basicConfig(filename='log.txt', level=logging.WARNING)
logger = logging.getLogger('main')

def main():
  logger.info("Start")
  prepare_data(9)
  show_nets('test',1)

if __name__ == '__main__':
  main()
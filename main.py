
from gravi.research.data import test
import logging

logging.basicConfig(filename='log.txt', level=logging.WARNING)
logger = logging.getLogger('main')

def main():
  logger.info("Start")
  test()

if __name__ == '__main__':
  main()
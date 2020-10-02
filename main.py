
print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))
from gravi.research.data import DataCreator

def main():
  print("Main menu")
  d = DataCreator()

if __name__ == '__main__':
  main()
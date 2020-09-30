from .data import DataCreator

def main():
  print("Begin researching!")

  creator = DataCreator()
  d = creator.create_data(1)

  

  print("End researching!")

if __name__ == '__main__':
  main()
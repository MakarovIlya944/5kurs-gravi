from gravi.research.main import learn


def research() -> None:
  baseConfig = {

  }

  # learn
  with open(f'data/{dataset_name}/0_in', 'r') as f:
    i = len(f.readlines())
  with open(f'data/{dataset_name}/0_out', 'r') as f:
    o = len(f.readlines())
  config_name = args['config']
  learn(i, o, dataset_name, config_name)
  # predict
  # inspect




if __name__ == '__main__':
  research()
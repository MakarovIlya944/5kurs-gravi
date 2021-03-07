from .data import  DataReader, Configurator
from paint import heatmaps

def show_net(dataset_config, dataset_name, index=0, save_image=False):
    label = f'{dataset_name}_{dataset_config}_{index}'
    dataset_config = Configurator.get_dataset_config(dataset_config)
    r_x = dataset_config['receptors']['x']
    r_y = dataset_config['receptors']['y']
    r = (dataset_config['receptors']['y']['n'],dataset_config['receptors']['x']['n'])
    r_x = range(r_x['l'],r_x['r'],(r_x['r'] - r_x['l']) // r_x['n'])
    r_y = range(r_y['l'],r_y['r'],(r_y['r'] - r_y['l']) // r_y['n'])

    X,Y,C = DataReader.read_one('data/' + dataset_name, index, out_format='tensor',shape='default')
    s = dataset_config['net']['count']
    Y = Y.detach().numpy().reshape(s).reshape((s[0],s[2]))

    filename = None
    if save_image:
        filename = label + '.png'
    heatmaps({'x': r_x, 'y':r_y, 'kx': 1, 'ky': 10},Y,None,None,label=label,save_filename=filename)

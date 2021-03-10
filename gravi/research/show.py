from gravi.reverse.builder import complex_build
from gravi.research.main import predict_one
from .data import  DataReader, Configurator
from paint import heatmaps
from .paint import paint_solidity
import matplotlib.pyplot as plt
from matplotlib.pyplot import hist, legend, plot
from .data import DataReader
from mpl_toolkits.mplot3d import axes3d, Axes3D
from .validate import *
from gravi.reverse import net,min

def show_net(dataset_config, dataset_name, index=0, save_image=False):
    label = f'net_{dataset_name}_{dataset_config}_{index}'
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

def show_predict_net(model_config, model_name, dataset_config, dataset_name, index=0, save_image=False):
    label = f'predict_{dataset_name}_{dataset_config}_{index}'
    dataset_config = Configurator.get_dataset_config(dataset_config)
    r_x = dataset_config['receptors']['x']
    r_y = dataset_config['receptors']['y']
    r = (dataset_config['receptors']['y']['n'],dataset_config['receptors']['x']['n'])
    r_x = range(r_x['l'],r_x['r'],(r_x['r'] - r_x['l']) // r_x['n'])
    r_y = range(r_y['l'],r_y['r'],(r_y['r'] - r_y['l']) // r_y['n'])

    X,Y,C = DataReader.read_one('data/' + dataset_name, index, out_format='tensor',shape='default')
    s = dataset_config['net']['count']
    d = {'config':model_config, 'name':model_name}
    predict_one(d,X,Y,s)
    Y = d['predicted'].reshape((s[0],s[2]))
    filename = None
    if save_image:
        filename = label + '.png'
    heatmaps({'x': r_x, 'y':r_y, 'kx': 1, 'ky': 10},Y,None,None,label=label,save_filename=filename)

def show_response(dataset_config, dataset_name, index=0, save_image=False):
    label = f'response_{dataset_name}_{dataset_config}_{index}'
    dataset_config = Configurator.get_dataset_config(dataset_config)
    r_x = dataset_config['receptors']['x']
    r_y = dataset_config['receptors']['y']
    r = (dataset_config['receptors']['y']['n'],dataset_config['receptors']['x']['n'])
    r_x = range(r_x['l'],r_x['r'],(r_x['r'] - r_x['l']) // r_x['n'])
    r_y = range(r_y['l'],r_y['r'],(r_y['r'] - r_y['l']) // r_y['n'])

    alpha=[0.1]
    gamma=None

    receptors = []
    for y in r_y:
      for x in r_x:
        receptors.append([float(x),float(y),0.0])
    receptors = np.asarray(receptors)

    trued = dataset_config['net']
    trued['values'] = {}

    X,Y,C = DataReader.read_one('data/' + dataset_name, index, out_format='tensor',shape='default')
    s = dataset_config['net']['count']
    for i,v in enumerate(Y[0]):
      trued['values'][(i%s[0],(i%(s[0]*s[1]))//s[0],i//(s[0]*s[1]))] = v
    trued = complex_build(params = trued)
    smile = min.Minimizator(net=trued, receptors=receptors, correct=trued, alpha=alpha, gamma=gamma, dryrun=True)
    net = smile.minimization()
    dGz = smile.solver.profile(net)
    trued = np.transpose(np.asarray(dGz).reshape(r))

    filename = None
    if save_image:
        filename = label + '.png'
    heatmaps({'x': r_x, 'y':r_y, 'kx': 1, 'ky': 1},trued,None,None,label=label,save_filename=filename)

def show_nets(name, params):
    x,y,c = DataReader.read_folder('data/' + name)

    fig, axs = plt.subplots(3, 3, sharex=True, sharey=True)
    for i in range(9):
        if c[i][1] == 1:
            _x = c[i][0]
            _y = c[i][2]
            m = [y[i][k*_x:(k+1)*_x] for k in range(_y)]
            j = i // 3
            k = i % 3
            axs[j, k].matshow(m,vmax=10,vmin=0)
            axs[j, k].set_title(str(i))
    plt.tight_layout()
    plt.show()

def show_3d(name):
    x,y,c = DataReader.read_folder('data/' + name)

    dx, dy, dz = 5, 5, 5
    for i in range(len(c)):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            m = max(y[i])
            for j, v in enumerate(y[i]):
                ax.scatter(j % c[i][0] * dx, j % (c[i][0] * c[i][1]) // c[i][1] * dy, j // (c[i][0] * c[i][1])  * -dz, marker='o', c=[[0.1,0.1,v/m]])

            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')

            plt.show()

def show_loss(name:str, save_image):
    if not name:
        raise Exception('Filename empty')
    with open(name, 'r') as f:
        ll = f.readlines()
        y_t = []
        y_v = []
        x = []
        j = 0
        for l in ll:
            i = l.index('train:')
            l = l[i+6:].split(' ')
            y_t.append(float(l[0]))

            i = l[1].index('val:')
            y_v.append(float(l[1][i+4:]))
            x.append(j)
            j+=1
    plot(x,y_t, label='train_loss')
    plot(x,y_v, label='val_loss')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
            ncol=2, mode="expand", borderaxespad=0.)
    if save_image:
        name = name.replace('.log','.png')
        plt.savefig(f'loss_{name}',dpi=300)
    else:
        plt.show()

def show_stat(dataset_config, dataset, mode, is_save):
    res = calc_stat(dataset, mode)
    dataset_config = Configurator.get_dataset_config(dataset_config)
    def_net = dataset_config['net']
    r_y = range(def_net['left'][0],def_net['right'][0],(def_net['right'][0] - def_net['left'][0]) // def_net['count'][0])
    r_x = range(def_net['left'][2],def_net['right'][2],(def_net['right'][2] - def_net['left'][2]) // def_net['count'][2])
    
    paint_solidity(res, {'r_x': r_x, 'r_y':r_y}, is_save)

def show_variation(dataset, save_image):
    A = count_net_variation(dataset)

    x = [A[k] for k in A] 
    hist(x,bins=len(x))
    filename = None
    if save_image:
        filename = f'{dataset}_variation.png'
    if filename:
        plt.savefig(filename,dpi=300)
    else:
        plt.show()

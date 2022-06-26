import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--by', default='ep_num', type=str, help='x-axis [ep_num, timesteps]')

x = parser.parse_args().by
if x not in ['ep_num', 'timesteps']:
    exit('Wrong arguments, choose between [ep_num, timesteps]')

path = './logs/train.npz'
log = np.load(path)
bounds = log['bounds']
my_labels = ['foot', '_Hidden', 'thigh', '_Hidden', 'leg', '_Hidden']
bounds_df = pd.DataFrame(bounds, index=log[x])
cmap = mpl.colors.ListedColormap(['red', 'green', 'blue'])
ax = bounds_df.plot(colormap=cmap)
handles, _ = ax.get_legend_handles_labels()
my_handles = [handles[i] for i in range(-2,4)]
ax.legend(labels=my_labels, handles=my_handles, fontsize='large')
ax.set_xlabel('timesteps' if x =='timesteps' else 'episodes')
ax.set_ylabel('mass (kg)')
plt.grid()
plt.show()

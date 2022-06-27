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

path = './train.npz'
log = np.load(path)
bounds = log['bounds']
my_labels = ['foot', '_Hidden', 'thigh', '_Hidden', 'leg', '_Hidden']
bounds_df = pd.DataFrame(bounds, index=log[x])
cmap = mpl.colors.ListedColormap(['red', 'green', 'blue'])
ax = bounds_df.plot(colormap=cmap)
handles, _ = ax.get_legend_handles_labels()
my_handles = [handles[i] for i in range(-2,4)]
ax.legend(labels=my_labels, handles=my_handles, fontsize='large', loc='upper right', bbox_to_anchor=(1, 0.9))
ax.set_xlabel('timesteps' if x =='timesteps' else 'episodes', fontsize='large')
ax.set_ylabel('mass (kg)', fontsize='large')
plt.grid()
plt.savefig('bounds_proportional.pdf', format='pdf')

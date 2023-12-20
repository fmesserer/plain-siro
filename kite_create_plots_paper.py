import os
import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
import seaborn as sns
from seaborn.palettes import color_palette

from kite_plotutils import plotKitePositionInAngleSpace
from plotutils_SIRO import plotContractionCompare
from latexify import latexify
latexify(fig_width=3.5)

# folder where plots will be saved
plot_folder = 'kite_plots_paper/'
if not os.path.exists(plot_folder): os.makedirs(plot_folder)

# folder where result files are saved
# (results files are created by kite_run.py)
res_folder = 'kite_results_paper/'

# plot trajectories for the following result file
res_file_main = 'kite_res_sigma-1.npy'

# plot SIRO contraction rates for the following result files
# followed by corresponding linestyles and labels
res_files_contr = ['kite_res_sigma-2.npy', 'kite_res_sigma-1.npy', 'kite_res_sigma-o5.npy']
line_style = [':', '--', '-']
labels=[ r'$\sigma = 2$', r'$\sigma = 1$', r'$\sigma = 0.5$']

# load result files
res = np.load(res_folder + res_file_main, allow_pickle=True)[()]

# set color scheme
color_palette = sns.color_palette('muted')
colors = color_palette.copy()
colors[1] = color_palette[3]
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors) 

# extract results
params = res['params']
Xnom = res['traj_nom']['X']
Unom = res['traj_nom']['U']
X_ol = res['traj_rol']['X']
U_ol = res['traj_rol']['U']
P_ol = res['traj_rol']['P']
X_cl = res['traj_rcl']['X']
U_cl = res['traj_rcl']['U']
P_cl = res['traj_rcl']['P']
K_cl = res['traj_rcl']['K']

#%% plot nominal trajectory
plotKitePositionInAngleSpace(params, Xnom, Unom, title='(a) nominal')
plt.legend(loc='best', bbox_to_anchor=(0.5, 0.5, 0.35, 0.5), borderpad=0.2)
plt.title('(a) nominal')
plt.tight_layout(pad=0.2)
plt.savefig(plot_folder + 'kite_nom_pos.pdf')#,  bbox_inches='tight', pad_inches=0)

#%% plot open loop trajectory
plotKitePositionInAngleSpace(params, X_ol, U_ol, P_ol, title='(b) open loop robust')
plt.title('(b) open loop robust')
plt.tight_layout(pad=0.2)
plt.savefig(plot_folder + 'kite_rol_pos.pdf')

#%% plot closed loop trajectory
plotKitePositionInAngleSpace(params, X_cl, U_cl, P_cl, title='(c) closed loop robust')
plt.title('(c) closed loop robust')
plt.tight_layout(pad=.2)
plt.savefig(plot_folder + 'kite_rcl_pos.pdf')

#%% compare contraction rates
# load iteration histories
it_hists = [[]] * len(res_files_contr)
for i, res_file in enumerate(res_files_contr):
    res = np.load(res_folder + res_file, allow_pickle=True)[()]
    it_hists[i] = res['SIRO_hist']
# plot and save
plotContractionCompare(it_hists, ls=line_style, labels=labels)
plt.tight_layout(pad=0.2)
plt.savefig(plot_folder + 'kite_contraction.pdf',  bbox_inches='tight', pad_inches=0.05)

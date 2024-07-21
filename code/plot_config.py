import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from matplotlib.colors import ListedColormap
import colorsys

# colors = mpl.cycler(color=["c", "m", "y", "r", "g", "b", "k"]) 
# color_cycle = ["#4DBBD5FF", "#E64B35FF", "#00A087FF", "#70699e", "#F39B7FFF", "#3C5488FF", "#7E6148FF","#DC0000FF",  "#91D1C2FF", "#B09C85FF", "#923a3a", "#8491B4FF"]
# color_cycle = ["#B65655FF", "#5471abFF", "#6aa66eFF", "#A66E6AFF", "#e3a13aFF", "#7a2c29FF", "#253a6aFF", "#8b9951FF"]
color_cycle = ["#b5423dFF", "#405977FF", "#616c3aFF", "#e3a13aFF", "#7a2c29FF", "#253a6aFF", "#8b9951FF"]

# Function to lighten a color
def lighten_color(color, amount=0.3):
    # Convert color from hexadecimal to RGB
    r, g, b, a = tuple(int(color[i:i+2], 16) for i in (1, 3, 5, 7))
    # Convert RGB to HLS
    h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
    # Lighten the luminance component
    l = min(1, l + amount)
    # Convert HLS back to RGB
    r, g, b = tuple(round(c * 255) for c in colorsys.hls_to_rgb(h, l, s))
    # Convert RGB back to hexadecimal
    new_color = f"#{r:02x}{g:02x}{b:02x}{a:02x}"
    return new_color

color_cycle_light = [lighten_color(color, 0.3) for color in color_cycle]
# color_cycle_light = [color[:-2] + "60" for color in color_cycle]
colors = mpl.cycler(mfc=color_cycle_light, color=color_cycle, markeredgecolor=color_cycle)

mpl.rc('axes', prop_cycle=colors)
# mpl.rc('axes', grid=True, edgecolor='k', prop_cycle=colors)
# mpl.rcParams['axes.prop_cycle'] = colors
# mpl.rcParams['lines.markeredgecolor'] = 'C'

mpl.rcParams['font.family'] = 'sans-serif'  # 'Helvetica'
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams["xtick.direction"] = "in" # 'out'
mpl.rcParams["ytick.direction"] = "in"
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['ytick.minor.width'] = 1.5
mpl.rcParams['lines.markersize'] = 10
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['lines.linewidth'] = 1.5
# plt.rcParams['lines.markeredgecolor'] = 'k'
mpl.rcParams['lines.markeredgewidth'] = 1.5
mpl.rcParams['figure.dpi'] = 130
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = True

SMALL_SIZE = 14
MEDIUM_SIZE = 18  #default 10
LARGE_SIZE = 24
# MARKER_SIZE = 10

plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=LARGE_SIZE+2)  # fontsize of the axes title
plt.rc('axes', labelsize=LARGE_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=LARGE_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=LARGE_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc('figure', titlesize=LARGE_SIZE)  # fontsize of the figure title

# def data_plot(x, y, marker, label, alpha=1, linewidth=1, loglog=True, markeredgecolor='black'):
#     if loglog:
#         plt.loglog(x, y, marker, label=label, linewidth=linewidth, markeredgecolor=markeredgecolor, markeredgewidth=0.5, alpha=alpha)
#     else:
#         plt.plot(x, y, marker, label=label, linewidth=linewidth, markeredgecolor=markeredgecolor, markeredgewidth=0.5, alpha=alpha)

from scipy.optimize import curve_fit
from math import ceil, floor, log, exp

def linear_loglog_fit(x, y, verbose=False):
    # Define the linear function
    def linear_func(x, a, b):
        return a * x + b

    log_x = np.array([log(n) for n in x])
    log_y = np.array([log(cost) for cost in y])
    # Fit the linear function to the data
    params, covariance = curve_fit(linear_func, log_x, log_y)
    # Extract the parameters
    a, b = params
    # Predict y values
    y_pred = linear_func(log_x, a, b)
    # Print the parameters
    if verbose: print('Slope (a):', a, 'Intercept (b):', b)
    exp_y_pred = [exp(cost) for cost in y_pred]

    return exp_y_pred, a, b

def plot_fit(ax, x, y, var='t', x_offset=1.07, y_offset=1.0, label='', ext_x=[], verbose=True):
    y_pred_em, a_em, b_em = linear_loglog_fit(x, y)
    if verbose: print(f'a_em: {a_em}; b_em: {b_em}')
    text_a_em = "{:.2f}".format(round(abs(a_em), 4))

    if ext_x != []: x = ext_x
    y_pred_em = [exp(cost) for cost in a_em*np.array([log(n) for n in x]) + b_em]
    if label =='':
        ax.plot(x, y_pred_em, 'k--', linewidth=2)
    else:
        ax.plot(x, y_pred_em, 'k--', linewidth=2, label=label)
    ax.annotate(r'$O(%s^{%s})$' % (var, text_a_em), xy=(x[-1], np.real(y_pred_em)[-1]), xytext=(x[-1]*x_offset, np.real(y_pred_em)[-1]*y_offset))

def ax_set_text(ax, x_label, y_label, title=None, legend='best', xticks=None, yticks=None, grid=None, log=''):
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title: ax.set_title(title)
    if legend: ax.legend(loc=legend)

    if xticks is not None: ax.set_xticks(xticks)
    if yticks is not None: ax.set_yticks(yticks)
    if grid: ax.grid()  

    if log == 'x': 
        ax.set_xscale('log')
    elif log == 'y':
        ax.set_yscale('log')
    elif log == 'xy':
        ax.loglog()
    else:
        pass

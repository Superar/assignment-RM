import matplotlib.lines as mlines
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import shape
import pandas as pd

FILE = 'EDA novo/results_11_06_2021_05_25_49.csv'


def linear_regression_3d(x, y, z):
    bias = np.ones(shape=x.shape)
    X = np.vstack([x, y, bias]).T
    Y = z.values

    # b = (X^T X)^-1 X^T Y
    params = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)
    return params


def linear_regression(x, y):
    bias = np.ones(shape=x.shape)
    X = np.vstack([x, bias]).T
    Y = y.values

    try:
        params = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)
    except np.linalg.LinAlgError:
        params = np.zeros(2)
    return params


def plot_linear_regression(df, ax, fixed_value,
                           fixed_attr, variable,
                           marker='o', marker_c='b',
                           regr_line_c='r', regr_marker_c='r'):
    rows = (df[fixed_attr] == fixed_value) & \
        (df['runtime'] < 100) & \
        (df['runtime'] > 0)
    prob_df = df.loc[rows, :]
    x, y = prob_df[variable], prob_df['runtime']
    ax.set_yscale('log')
    ax.scatter(x, y, marker=marker, c=marker_c, alpha=0.3)

    regr_params = linear_regression(x, np.log10(y))
    regr_x = x.sort_values().unique()
    regr_x = np.vstack([regr_x, np.ones(regr_x.shape)])
    regr_y = np.clip(np.power(10, np.dot(regr_params.T, regr_x)), 0, 100)
    ax.scatter(regr_x[0, :], regr_y, marker=',', c=regr_marker_c)

    regr_line_x = np.linspace(df[variable].min(), df[variable].max(), 1000)
    regr_line_x = np.vstack([regr_line_x, np.ones(regr_line_x.shape)])
    regr_line_y = np.clip(
        np.power(10, np.dot(regr_params.T, regr_line_x)), 0, 100)
    ax.plot(regr_line_x[0, :], regr_line_y, c=regr_line_c)

    # Calculate residuals
    residuals = y - \
        np.clip(np.power(10, np.dot(regr_params.T,
                np.vstack([x, np.ones(x.shape)]))), 0, 100)
    sst = np.square(y - y.mean()).sum()
    sse = np.square(residuals).sum()
    ssr = sst - sse
    r2 = (ssr / sst)*100 if sst > 0 else 0.0
    return r2


df = pd.read_csv(FILE, index_col=0)
clean_info = df['file'].str.lstrip('data_').str.rstrip('.in')
split_info = clean_info.str.split('-')

df[['Number of exams', 'Overlap probability', 'Seed']] = split_info.to_list()
df['Number of exams'] = df['Number of exams'].astype('int')
df['Overlap probability'] = df['Overlap probability'].astype('float')
df['Seed'] = df['Seed'].astype('int')

### LINEAR REGRESSION ###
df_no_timeout = df.loc[df['runtime'] != 100, :]
df_no_timeout = df_no_timeout.loc[df_no_timeout['runtime'] > 0, :]
df_no_timeout['log runtime'] = df_no_timeout['runtime'].apply(np.log10)

code1_runs = df_no_timeout.loc[df_no_timeout['algorithm'] == 'code1', :]
code2_runs = df_no_timeout.loc[df_no_timeout['algorithm'] == 'code2', :]

# code1
code1_params = linear_regression_3d(code1_runs['Overlap probability'],
                                    code1_runs['Number of exams'],
                                    code1_runs['log runtime'])
code1_surf_x, code1_surf_y = np.meshgrid(code1_runs['Overlap probability'],
                                         code1_runs['Number of exams'])
code1_surf_x, code1_surf_y = code1_surf_x.flatten(), code1_surf_y.flatten()
code1_surf_bias = np.ones(shape=code1_surf_x.shape)
code1_surf_z = np.dot(code1_params.T,
                      np.vstack([code1_surf_x, code1_surf_y, code1_surf_bias]))

# code2
code2_params = linear_regression_3d(code2_runs['Overlap probability'],
                                    code2_runs['Number of exams'],
                                    code2_runs['log runtime'])
code2_surf_x, code2_surf_y = np.meshgrid(code2_runs['Overlap probability'],
                                         code2_runs['Number of exams'])
code2_surf_x, code2_surf_y = code2_surf_x.flatten(), code2_surf_y.flatten()
code2_surf_bias = np.ones(shape=code2_surf_x.shape)
code2_surf_z = np.dot(code2_params.T,
                      np.vstack([code2_surf_x, code2_surf_y, code2_surf_bias]))

### PLOT IN LOG SPACE ###
fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': '3d'},
                               figsize=(15, 5))
plt.tight_layout(w_pad=7)

# code1
ax1.view_init(13, -36)
ax1.scatter(code1_runs['Overlap probability'],
            code1_runs['Number of exams'],
            code1_runs['log runtime'])
ax1.plot_trisurf(code1_surf_x, code1_surf_y, np.clip(code1_surf_z, code1_surf_z.min(), 2),
                 alpha=0.3, shade=False, color='r')
ax1.zaxis.set_major_formatter(mtick.FuncFormatter(lambda z, _: f'$10^{{{z:.0f}}}$'))
ax1.set_title('code1', y=1, pad=0)
ax1.set_xlabel('Overlap probability')
ax1.set_ylabel('Number of exams')
ax1.set_zlabel('Run time')

# code2
ax2.view_init(13, -36)
ax2.scatter(code2_runs['Overlap probability'],
            code2_runs['Number of exams'],
            code2_runs['log runtime'])
ax2.plot_trisurf(code2_surf_x, code2_surf_y, np.clip(code2_surf_z, code2_surf_z.min(), 2),
                 alpha=0.3, shade=False, color='r')
ax2.zaxis.set_major_formatter(mtick.FuncFormatter(lambda z, _: f'$10^{{{z:.0f}}}$'))
ax2.set_title('code2', y=1, pad=0)
ax2.set_xlabel('Overlap probability')
ax2.set_ylabel('Number of exams')
ax2.set_zlabel('Runtime')

plt.savefig('img/linear_regression_log.pdf', format='pdf')

### PLOT IN EXP SPACE ###

fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': '3d'},
                               figsize=(15, 5))
plt.tight_layout(w_pad=2)

# code1
ax1.view_init(13, -36)
ax1.scatter(code1_runs['Overlap probability'],
            code1_runs['Number of exams'],
            code1_runs['runtime'])
code1_surf_z_exp = np.clip(np.power(10, code1_surf_z), 0, 100)
ax1.plot_trisurf(code1_surf_x, code1_surf_y, code1_surf_z_exp,
                 alpha=0.3, shade=False, color='r')
ax1.set_title('code1', y=1, pad=0)
ax1.set_xlabel('Overlap probability')
ax1.set_ylabel('Number of exams')
ax1.set_zlabel('Runtime')

# code2
ax2.view_init(13, -36)
ax2.scatter(code2_runs['Overlap probability'],
            code2_runs['Number of exams'],
            code2_runs['runtime'])
code2_surf_z_exp = np.clip(np.power(10, code2_surf_z), 0, 100)
ax2.plot_trisurf(code2_surf_x, code2_surf_y, code2_surf_z_exp,
                 alpha=0.3, shade=False, color='r')

ax2.set_title('code2', y=1, pad=0)
ax2.set_xlabel('Overlap probability')
ax2.set_ylabel('Number of exams')
ax2.set_zlabel('Runtime')

plt.savefig('img/linear_regression.pdf', format='pdf')

### GRID LINEAR REGRESSION - FIXED PROBABILITY ###
code1_df = df.loc[df['algorithm'] == 'code1']
code2_df = df.loc[df['algorithm'] == 'code2']

ncols = 3
probs = sorted(df['Overlap probability'].unique())
naxs_last_row = len(probs) % ncols
nrows = len(probs) // ncols
nrows = nrows + 1 if naxs_last_row else nrows

fig, axs = plt.subplots(nrows, ncols, figsize=(15, 10))
for i in range(ncols - naxs_last_row):
    axs[-1][-i-1].remove()

for i, prob in enumerate(probs):
    ax = axs[i // ncols][i % ncols]
    code1_r2 = plot_linear_regression(code1_df, ax, prob,
                                      'Overlap probability',
                                      'Number of exams',
                                      marker='d', marker_c='k',
                                      regr_line_c='m', regr_marker_c='m')
    code2_r2 = plot_linear_regression(code2_df, ax, prob,
                                      'Overlap probability',
                                      'Number of exams',
                                      marker_c='g')
    ax.set_xlabel('Number of exams')
    ax.set_ylabel('Run time')
    ax.set_title(f'''Probability {prob*100:.2f}%
    code1.c - $R^2$ = {code1_r2:.2f}%
    code2.c - $R^2$ = {code2_r2:.2f}%''')

# Add legend
magenta_patch = mlines.Line2D([], [], c='m', marker='s',
                              label='Linear Regression - code1.c')
red_patch = mlines.Line2D([], [], c='r', marker='s',
                          label='Linear Regression - code2.c')
black_diamond = mlines.Line2D([], [], c='k', marker='d', linestyle='None',
                              alpha=0.3, label='Observations - code1.c')
green_circle = mlines.Line2D([], [], c='g', marker='o', linestyle='None',
                             alpha=0.3, label='Observations - code2.c')
fig.legend(handles=[magenta_patch, red_patch,
                    black_diamond, green_circle],
           loc='lower right', bbox_to_anchor=(0.87, 0.15))

fig.tight_layout()
fig.savefig('img/linear_regression_grid_probability_fixed.png', format='png')
fig.savefig('img/linear_regression_grid_probability_fixed.pdf', format='pdf')

### GRID LINEAR REGRESSION - FIXED NUMBER OF EXAMS ###
ncols = 6
nums = sorted(df['Number of exams'].unique())
naxs_last_row = len(nums) % ncols
nrows = len(nums) // ncols
nrows = nrows + 1 if naxs_last_row else nrows

fig, axs = plt.subplots(nrows, ncols, figsize=(20, 10))
for i in range(ncols - naxs_last_row):
    axs[-1][-i-1].remove()

for i, num in enumerate(nums):
    ax = axs[i // ncols][i % ncols]
    code1_r2 = plot_linear_regression(code1_df, ax, num,
                                      'Number of exams',
                                      'Overlap probability',
                                      marker='d', marker_c='k',
                                      regr_line_c='m', regr_marker_c='m')
    code2_r2 = plot_linear_regression(code2_df, ax, num,
                                      'Number of exams',
                                      'Overlap probability',
                                      marker_c='g')

    ax.set_xlabel('Probability')
    ax.set_ylabel('Run time')
    ax.set_title(f'''{num} exams
    code1.c - $R^2$ = {code1_r2:.2f}%
    code2.c - $R^2$ = {code1_r2:.2f}%''')

# Add legend
magenta_patch = mlines.Line2D([], [], c='m', marker='s',
                              label='Linear Regression - code1.c')
red_patch = mlines.Line2D([], [], c='r', marker='s',
                          label='Linear Regression - code2.c')
black_diamond = mlines.Line2D([], [], c='k', marker='d', linestyle='None',
                              alpha=0.3, label='Observations - code1.c')
green_circle = mlines.Line2D([], [], c='g', marker='o', linestyle='None',
                             alpha=0.3, label='Observations - code2.c')
fig.legend(handles=[magenta_patch, red_patch,
                    black_diamond, green_circle],
           loc='lower right', bbox_to_anchor=(0.8, 0.05))

fig.tight_layout()
fig.savefig('img/linear_regression_grid_num_exams_fixed.png', format='png')
fig.savefig('img/linear_regression_grid_num_exams_fixed.pdf', format='pdf')

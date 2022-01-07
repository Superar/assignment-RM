from statsmodels.graphics.gofplots import qqplot
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.sandbox.stats.multicomp import MultiComparison
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

FILE = 'results-milestone3/results_12_31_2021_14_08_35.csv'
IGNORE_TIMEOUT = True

if IGNORE_TIMEOUT:
    print('IGNORING TIMEOUT ENTRIES')

print('Reading data')
df = pd.read_csv(FILE, index_col=0)
clean_info = df['file'].str.lstrip('data_').str.rstrip('.in')
split_info = clean_info.str.split('-')

df[['n_exams', 'prob', 'seed']] = split_info.to_list()
df['n_exams'] = df['n_exams'].astype('int')
df['prob'] = df['prob'].astype('float')
df['seed'] = df['seed'].astype('int')
df['log_runtime'] = df['runtime'].apply(np.log10)
df = df.loc[df['runtime'] > 0, :]
if IGNORE_TIMEOUT:
    df = df.loc[df['runtime'] < 100, :]
print('Data loaded')

# ANOVA
print('Running ANOVA')
model = ols('log_runtime ~ C(algorithm)*n_exams*prob', data=df).fit()
anova_results = anova_lm(model, typ=2)
anova_results['rejected'] = anova_results['PR(>F)'] < 0.05
print(anova_results.to_string())
print('\n')
print(anova_results.to_latex(
    columns=['sum_sq', 'df', 'F', 'PR(>F)'],
    header=['Sum of Squares', 'Degrees of Freedom', 'F', 'PR(>F)'],
    float_format='{:.4e}'.format,
    caption='ANOVA results'))

# Residuals-vs-fitted plot
print('Plotting residuals-vs-fitted')
plt.scatter(model.fittedvalues, model.resid_pearson, c='None', edgecolors='k')
unique_fittedvalues = model.fittedvalues.unique()
plt.plot(unique_fittedvalues,
         np.repeat(0, len(unique_fittedvalues)),
         'k:', lw=1)
lowess_resid = lowess(model.resid_pearson,
                      model.fittedvalues,
                      return_sorted=True)
plt.plot(lowess_resid[:, 0], lowess_resid[:, 1], c='r', lw=0.5)
plt.title('Residuals vs Fitted')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
filename = 'residuals_vs_fitted_clean.png' if IGNORE_TIMEOUT else 'residuals_vs_fitted.png'
plt.savefig(f'img/{filename}')

# Normal Q-Q plot
print('Plotting Normal Q-Q')
fig = qqplot(model.resid_pearson, line='45')
fig.title = 'Normal Q-Q'
filename = 'anova_qq_clean.png' if IGNORE_TIMEOUT else 'anova_qq.png'
fig.savefig(f'img/{filename}')

# Post-hoc analysis using TukeyHSD test
print('Preparing TukeyHSD data')
groups_df = df.groupby(by=['algorithm', 'n_exams', 'prob']).groups
data = list()
groups = list()

for g in groups_df:
    g_data = df.loc[groups_df[g], 'runtime'].values
    group_label = ':'.join((str(v) for v in g))
    groups.extend(np.repeat(group_label, len(g_data)))
    data.extend(g_data)
print('Running TukeyHSD')
mod1 = MultiComparison(np.array(data), groups)
tukeyhsd_out = mod1.tukeyhsd(alpha=0.05)

# Convert TukeyHSD result to pandas DataFrame
tukeyhsd_html = tukeyhsd_out.summary().as_html()
tukeyhsd_df = pd.read_html(tukeyhsd_html, header=0)[0]
filename = 'tukeyHSD_results_complete_clean.csv' if IGNORE_TIMEOUT else 'tukeyHSD_results_complete.csv'
tukeyhsd_df.to_csv(f'results-milestone3/{filename}', index=False)

# Select relevant rows
group1_cols = tukeyhsd_df['group1'].str.split(':', expand=True)
tukeyhsd_df[['group1_algorithm', 'group1_exams', 'group1_prob']] = group1_cols
group2_cols = tukeyhsd_df['group2'].str.split(':', expand=True)
tukeyhsd_df[['group2_algorithm', 'group2_exams', 'group2_prob']] = group2_cols
tukeyhsd_code1 = tukeyhsd_df['group1_algorithm'] == 'code1'
tukeyhsd_code2 = tukeyhsd_df['group2_algorithm'] == 'code2'
rejected = tukeyhsd_df['reject']
same_n_exams = tukeyhsd_df['group1_exams'] == tukeyhsd_df['group2_exams']
same_prob = tukeyhsd_df['group1_prob'] == tukeyhsd_df['group2_prob']
same_parameters = same_n_exams & same_prob

relevant_pairs = tukeyhsd_code1 & tukeyhsd_code2 & same_parameters
filtered_pairs = tukeyhsd_df.loc[relevant_pairs, ['group1', 'group2', 'meandiff',
                                                  'p-adj', 'lower', 'upper', 'reject']]
filename = 'tukeyHSD_results_filtered_clean.csv' if IGNORE_TIMEOUT else 'tukeyHSD_results_filtered.csv'
filtered_pairs.to_csv(f'results-milestone3/{filename}',
                      index=False)

# Relation plot -- All three factors
fig, ax = plt.subplots(1, 1)
ax.set_yscale('log')
for i, p in enumerate(df['prob'].unique()):
    c = mcolors.TABLEAU_COLORS[list(mcolors.TABLEAU_COLORS)[i]]
    for a in ['code1', 'code2']:
        plot_data = df.query(f'prob == {p} & algorithm == "{a}"')
        mean_time = plot_data.groupby(by='n_exams').mean()
        if a == 'code1':
            ax.plot(mean_time.index, mean_time['runtime'],
                    label=f'{a} - {p}', c=c)
        if a == 'code2':
            ax.plot(mean_time.index, mean_time['runtime'],
                    label=f'{a} - {p}',
                    linestyle='--', c=c)
ax.legend(title='Overlap probability')
ax.set_xlabel('Number of exams')
ax.set_ylabel('Mean run time')
filename = 'three_factor_relation_plot_clean.png' if IGNORE_TIMEOUT else 'three_factor_relation_plot.png'
plt.savefig(f'./img/{filename}')

# Relation plot -- Two factors at a time
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_yscale('log')
ax2.set_yscale('log')
for a in df['algorithm'].unique():
    plot_data = df.query(f'algorithm == "{a}"')
    mean_time_n_exams = plot_data.groupby(by='n_exams').mean()
    mean_time_prob = plot_data.groupby(by='prob').mean()

    ax1.plot(mean_time_n_exams.index,
             mean_time_n_exams['runtime'], label=f'{a}')
    ax2.plot(mean_time_prob.index, mean_time_prob['runtime'], label=f'{a}')
ax1.legend(title='Algorithm')
ax1.set_title('C(algorithm):n_exams')
ax1.set_xlabel('Number of exams')
ax1.set_ylabel('Mean run time')

ax2.set_title('C(algorithm):prob')
ax2.set_xlabel('Overlap probability')
fig.tight_layout()
filename = 'two_factor_relation_plot_clean.png' if IGNORE_TIMEOUT else 'two_factor_relation_plot.png'
plt.savefig(f'./img/{filename}')

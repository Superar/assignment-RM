from statsmodels.graphics.gofplots import qqplot
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.sandbox.stats.multicomp import MultiComparison
from statsmodels.stats import anova
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILE = 'results-milestone3/results_12_31_2021_14_08_35.csv'

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
plt.savefig('img/residuals_vs_fitted.png')

# Normal Q-Q plot
print('Plotting Normal Q-Q')
fig = qqplot(model.resid_pearson, line='45')
fig.title = 'Normal Q-Q'
fig.savefig('img/anova_qq.png')

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
group1_cols = tukeyhsd_df['group1'].str.split(':', expand=True)
tukeyhsd_df[['group1_algorithm', 'group1_exams', 'group1_prob']] = group1_cols
group2_cols = tukeyhsd_df['group2'].str.split(':', expand=True)
tukeyhsd_df[['group2_algorithm', 'group2_exams', 'group2_prob']] = group2_cols
tukeyhsd_df.to_csv('results-milestone3/tukeyHSD_results_complete.csv',
                   index=False)

# Select relevant rows
tukeyhsd_code1 = tukeyhsd_df['group1_algorithm'] == 'code1'
tukeyhsd_code2 = tukeyhsd_df['group2_algorithm'] == 'code2'
rejected = tukeyhsd_df['reject']
same_n_exams = tukeyhsd_df['group1_exams'] == tukeyhsd_df['group2_exams']
same_prob = tukeyhsd_df['group1_prob'] == tukeyhsd_df['group2_prob']
same_parameters = same_n_exams & same_prob

relevant_pairs = tukeyhsd_code1 & tukeyhsd_code2 & same_parameters
filtered_pairs = tukeyhsd_df.loc[relevant_pairs, ['group1', 'group2', 'meandiff',
                                                  'p-adj', 'lower', 'upper', 'reject']]
filtered_pairs.to_csv('results-milestone3/tukeyHSD_results_filtered.csv',
                      index=False)

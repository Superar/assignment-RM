from statsmodels.graphics.gofplots import qqplot
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.sandbox.stats.multicomp import MultiComparison
from statsmodels.stats import anova
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILE = 'EDA-granularity/results_11_06_2021_05_25_49.csv'

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
print(anova_results)

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
print(tukeyhsd_out)
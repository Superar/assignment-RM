from statsmodels.stats.anova import anova_lm
from statsmodels.api import qqplot
from statsmodels.formula.api import ols
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.multicomp import MultiComparison
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data
df = pd.DataFrame([['A', 'A', 0.0952],
                   ['A', 'A', 0.0871],
                   ['A', 'A', 0.0969],
                   ['A', 'A', 0.1054],
                   ['A', 'A', 0.0812],
                   ['A', 'B', 0.1432],
                   ['A', 'B', 0.1343],
                   ['A', 'B', 0.1314],
                   ['A', 'B', 0.1443],
                   ['A', 'B', 0.1312],
                   ['B', 'A', 0.1382],
                   ['B', 'A', 0.1332],
                   ['B', 'A', 0.1482],
                   ['B', 'A', 0.1430],
                   ['B', 'A', 0.1483],
                   ['B', 'B', 0.1082],
                   ['B', 'B', 0.1032],
                   ['B', 'B', 0.1182],
                   ['B', 'B', 0.1130],
                   ['B', 'B', 0.1083],
                   ['C', 'A', 0.0966],
                   ['C', 'A', 0.1200],
                   ['C', 'A', 0.1152],
                   ['C', 'A', 0.1375],
                   ['C', 'A', 0.1298],
                   ['C', 'B', 0.1066],
                   ['C', 'B', 0.1100],
                   ['C', 'B', 0.1252],
                   ['C', 'B', 0.1275],
                   ['C', 'B', 0.1398]], columns=['system', 'prog', 'time'])

# Linear regression
model = ols('time ~ system*prog', data=df).fit()
# Two-way anova
anova_results = anova_lm(model, typ=2)

# Residuals-vs-fitted plot
# plt.scatter(model.fittedvalues, model.resid_pearson, c='None', edgecolors='k')
# plt.plot(model.fittedvalues.unique(), np.repeat(0, 6), 'k:', lw=1)
# lowess_resid = lowess(model.resid_pearson, model.fittedvalues, return_sorted=True)
# plt.plot(lowess_resid[:, 0], lowess_resid[:, 1], c='r', lw=0.5)
# plt.title('Residuals vs Fitted')
# plt.xlabel('Fitted values')
# plt.ylabel('Residuals')
# plt.savefig('residuals_vs_fitted.png')

# Normal Q-Q plot
# fig = qqplot(model.resid_pearson, line='45')
# fig.title = 'Normal Q-Q'
# fig.savefig('anova_qq.png')

# Post-hoc analysis using TukeyHSD test
groups_df = df.groupby(by=['system', 'prog']).groups
data = list()
groups = list()

for g in groups_df:
    g_data = df.loc[groups_df[g], 'time'].values
    groups.extend(np.repeat(':'.join(g), len(g_data)))
    data.extend(g_data)
mod1 = MultiComparison(np.array(data), groups)
tukeyhsd_out = mod1.tukeyhsd(alpha=0.05)
tukeyhsd_html = tukeyhsd_out.summary().as_html()
tukeyhsd_df = pd.read_html(tukeyhsd_html, header=0)[0]
group1_cols = tukeyhsd_df['group1'].str.split(':', expand=True)
tukeyhsd_df[['group1_system', 'group1_programmer']] = group1_cols
group2_cols = tukeyhsd_df['group2'].str.split(':', expand=True)
tukeyhsd_df[['group2_system', 'group2_programmer']] = group2_cols
print(tukeyhsd_df)

# rejected = tukeyhsd_df.loc[tukeyhsd_df['reject'], :]
# print(rejected)

# Relation plot
# fig = plt.figure()
# for p in df['prog'].unique():
#     mean_time = df.loc[df['prog'] == p, :].groupby(by='system').mean()
#     plt.plot(mean_time.index, mean_time['time'], label=p)
# plt.legend(title='Programmer')
# plt.xlabel('System')
# plt.ylabel('Mean of Time')
# plt.savefig('relation.png')

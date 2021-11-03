from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--results', '-r',
                    help='Results csv file',
                    type=Path,
                    default=Path('results.csv'))

args = parser.parse_args()

df = pd.read_csv(args.results, index_col=0)

# Get information from filenames
clean_info = df['file'].str.lstrip('data_').str.rstrip('.in')
split_info = clean_info.str.split('-')

df[['Number of exams', 'Overlap probability', 'Seed']] = split_info.to_list()
df['Number of exams'] = df['Number of exams'].astype('int')
df['Overlap probability'] = df['Overlap probability'].astype('float')
df['Seed'] = df['Seed'].astype('int')

#### Plots ####

# Timeout histogram
num_slots_data = df.groupby(by=['algorithm'])['slots'].value_counts()
num_slots_counts = num_slots_data.index.get_level_values('slots')

timeout_counts = num_slots_data[num_slots_counts == -1]
timeout_counts.index = timeout_counts.index.droplevel(1)
timeout_counts.name = 'Timeout'
non_timeout_counts = num_slots_data[num_slots_counts != -1]
non_timeout_counts = non_timeout_counts.groupby('algorithm').sum()
non_timeout_counts.name = 'Non-timeout'

timeout_df = pd.concat([timeout_counts, non_timeout_counts], axis=1)
timeout_df.index = ['Code 1', 'Code 2']
ax = timeout_df.plot.bar()
for p in ax.patches:
    ax.annotate(str(p.get_height()),
                (p.get_x() + (p.get_width() * 0.5), p.get_height() * 1.005),
                ha='center')
plt.tight_layout()
plt.savefig('img/timeout_hist.pdf', format='pdf')

# Grid scatter plot
code1_df = df.loc[df['algorithm'] == 'code1']
code2_df = df.loc[df['algorithm'] == 'code2']

fig, axs = plt.subplots(2, 3,
                        figsize=(15, 10))
fig.suptitle('Code 1', fontsize=16)

for i, prob in enumerate(df['Overlap probability'].unique()):
    ax = axs[i // 3][i % 3]
    prob_df = code1_df.loc[code1_df['Overlap probability'] == prob]
    ax.scatter(prob_df['Number of exams'], prob_df['runtime'],
               marker='o', alpha=0.3)
    ax.set_xlabel('Number of exams')
    ax.set_ylabel('Runtime')
    ax.set_title(f'Probability {prob}')
plt.savefig('img/grid_scatter_code1.pdf', format='pdf')

fig, axs = plt.subplots(2, 3,
                        figsize=(15, 10))
fig.suptitle('Code 2', fontsize=16)

for i, prob in enumerate(df['Overlap probability'].unique()):
    ax = axs[i // 3][i % 3]
    prob_df = code2_df.loc[code2_df['Overlap probability'] == prob]
    ax.scatter(prob_df['Number of exams'], prob_df['runtime'],
               marker='o', alpha=0.3)
    ax.set_xlabel('Number of exams')
    ax.set_ylabel('Runtime')
    ax.set_title(f'Probability {prob}')
plt.savefig('img/grid_scatter_code2.pdf', format='pdf')

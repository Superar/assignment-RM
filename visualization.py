from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
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
timeout_df.plot.bar()
plt.tight_layout()

# Average runtime by probability

runtime_data = df.groupby(by=['algorithm',
                              'Overlap probability'])['runtime'].mean()
code1_runtime = runtime_data['code1']
code1_runtime.name = 'Code 1'
code2_runtime = runtime_data['code2']
code2_runtime.name = 'Code 2'

runtime_df = pd.concat([code1_runtime, code2_runtime], axis=1)
runtime_df.plot.line(subplots=True)

plt.tight_layout()
plt.show()

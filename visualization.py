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
timeout_df.plot.bar()
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FILE = 'results_11_03_2021_04_41_08.csv'
EPS = 1e-4

df = pd.read_csv(FILE, index_col=0)
code1_df = df.loc[df['algorithm'] == 'code1', :]
code1_df.set_index('file', inplace=True)
code2_df = df.loc[df['algorithm'] == 'code2', :]
code2_df.set_index('file', inplace=True)

interval = np.abs(code1_df['runtime'] - code2_df['runtime'])
clean_idx = code1_df[interval >= EPS].index
clean_df = df.loc[df['file'].isin(clean_idx), :]
clean_df.to_csv(f'clean_{EPS}_{FILE}')

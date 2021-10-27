from subprocess import PIPE, run
from pathlib import Path
from sys import platform
import itertools
import argparse
import random
import pandas as pd

TIMEOUT = 5
NUM_EXAMS = 50
NUM_SEEDS = 3

def generate_data(seeds):
    n_exams = list(range(0, NUM_EXAMS+10, 10))
    # Fix limits
    n_exams[0] = 2

    probabilities = [2**(i+1) / 100 for i in range(6)] # TODO: Define final probabilities

    python = 'python' if platform == 'win32' else 'python3.9'
    for n, p, s in itertools.product(n_exams, probabilities, seeds):
        run([python,
             'material/gen.py',
             str(n), str(p), str(s),
             f'data/data_{n}-{p}-{s}.in'])


def run_tests(data_path):
    results = {
        'file': list(),
        'algorithm': list(),
        'slots': list(),
        'runtime': list()
    }
    for file_ in data_path.iterdir():
        seed = file_.stem.split('-')[-1]

        code1 = 'code1.exe' if platform == 'win32' else 'code1'
        code2 = 'code2.exe' if platform == 'win32' else 'code2'

        result_code1 = run([f'material/{code1}', seed, str(TIMEOUT), str(file_)], stdout=PIPE)
        slots_code1 = int(result_code1.stdout.split()[0])
        runtime_code1 = float(result_code1.stdout.split()[1])

        result_code2 = run([f'material/{code2}', seed, str(TIMEOUT), str(file_)], stdout=PIPE)
        slots_code2 = int(result_code2.stdout.split()[0])
        runtime_code2 = float(result_code2.stdout.split()[1])

        results['file'].append(file_.name)
        results['file'].append(file_.name)
        results['algorithm'].append('code1')
        results['algorithm'].append('code2')
        results['slots'].append(slots_code1)
        results['slots'].append(slots_code2)
        results['runtime'].append(runtime_code1)
        results['runtime'].append(runtime_code2)
    
    df = pd.DataFrame(results)
    df.to_csv('results.csv')
    

def main(args):
    if args.seeds:
        seeds = args.seeds
    else:
        seeds = [random.randint(1, 10000) for _ in range(NUM_SEEDS)] # TODO: Run with 30 seeds
    print(f'Seeds used: {" ".join(str(s) for s in seeds)}')

    if args.generate_data:
        generate_data(seeds)
    if args.data:
        run_tests(args.data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_data', '-g',
                        help='Generate data',
                        action='store_true')
    parser.add_argument('--seeds', '-s',
                        help='Fixed seeds',
                        nargs='+',
                        type=int)
    parser.add_argument('--data', '-d',
                        help='Fixed data directory',
                        type=Path)

    args = parser.parse_args()

    main(args)

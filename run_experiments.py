import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    speed = p.add_mutually_exclusive_group()
    speed.add_argument('--quick', action='store_true', help='Run with reduced experiment settings')
    speed.add_argument('--full', action='store_true', help='Run full settings')
    p.add_argument('--suite', choices=['all', 'primary', 'climate'], default='all')
    p.add_argument('--output', type=Path, default=None)
    p.add_argument('--data-path', type=Path, default=Path('data/engine_map.csv'))
    p.add_argument('--seeds', nargs='+', type=int, default=[42])
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--fold-limit', type=int, default=None)
    p.add_argument('--resume', action='store_true', help='Resume matching climate checkpoints; primary resumes automatically')
    p.add_argument('--primary-tasks', nargs='+', default=['baseline', 'severity', 'diversity'])
    p.add_argument('--variants', nargs='+', default=['candidate', 'adjusted', 'covariance', 'pca'])
    args = p.parse_args()
    if args.threads < 1:
        p.error('--threads must be positive')
    root = Path(__file__).resolve().parent
    output = (args.output or Path('outputs/quick' if args.quick else 'outputs/full')).expanduser().resolve()
    env = os.environ.copy()
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        env[name] = str(args.threads)
    env['MPLBACKEND'] = 'Agg'
    env['PYTHONUNBUFFERED'] = '1'
    common = ['--seeds', *map(str, args.seeds), '--data-path', str(args.data_path.expanduser().resolve())]
    if args.quick:
        common += ['--quick']
    if args.fold_limit is not None:
        common += ['--fold-limit', str(args.fold_limit)]
    jobs = []
    if args.suite in ('all', 'primary'):
        jobs.append(('primary', [sys.executable, '-m', 'experiments.primary',
                     '--output', str(output / 'primary'), '--threads', str(args.threads),
                     '--tasks', *args.primary_tasks, '--variants', *args.variants, *common]))
    if args.suite in ('all', 'climate'):
        jobs.append(('climate', [sys.executable, '-m', 'experiments.climate',
                     '--output', str(output / 'climate'), *common,
                     *(['--resume'] if args.resume else [])]))
    started = time.monotonic()
    for name, command in jobs:
        print(f'Running {name} experiments. Output folder: {output / name}', flush=True)
        subprocess.run(command, cwd=root, env=env, check=True)
    print(f'Finished in {(time.monotonic()-started)/60:.1f} minutes. Results: {output}', flush=True)


if __name__ == '__main__':
    main()

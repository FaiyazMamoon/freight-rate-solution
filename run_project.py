"""Run the reproducible solution while preserving the eight assessment inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def hashes(names):
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--load-models', action='store_true', help='Use trusted saved models instead of fitting final models')
    parser.add_argument('--skip-evaluation', action='store_true', help='Keep existing production backtests')
    parser.add_argument('--skip-tests', action='store_true', help='Keep the existing behavioral test result')
    args = parser.parse_args()
    original = json.loads((ROOT / 'analysis/source_manifest.json').read_text(encoding='utf-8'))['sha256']
    if hashes(original) != original:
        raise RuntimeError('Assessment inputs do not match the original source manifest')
    if args.skip_evaluation and not (ROOT / 'evaluation/production_results.json').exists():
        parser.error('--skip-evaluation requires existing evaluation/production_results.json')
    commands = []
    if not args.skip_tests:
        commands.append(['-m', 'unittest', 'discover', '-s', 'tests', '-v'])
    if not args.skip_evaluation:
        commands.append(['evaluate.py', '--metric-atol', '0.05'])
    commands.extend([
        ['train_predict.py'] + (['--load-models'] if args.load_models else []),
        ['score.py', '--predictions', 'validation_predictions.csv', '--december-predictions', 'december_predictions.csv'],
        ['reports/build_report.py'],
        ['validation/check_submission.py'],
    ])
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       JOBLIB_MULTIPROCESSING='0', PYTHONUTF8='1')
    log_dir = ROOT / 'run_logs'
    log_dir.mkdir(exist_ok=True)
    records = []
    for number, command in enumerate(commands, 1):
        print('Running ' + ' '.join(command), flush=True)
        log_path = log_dir / f'{number:02d}_{Path(command[0]).stem.replace("-", "")}.log'
        with log_path.open('w', encoding='utf-8') as stream:
            result = subprocess.run([sys.executable, *command], cwd=ROOT, env=environment,
                                    stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'Command failed ({result.returncode}); inspect {log_path.relative_to(ROOT)}')
        if hashes(original) != original:
            raise RuntimeError('An original input changed during ' + command[0])
        records.append(dict(command=command, exit_code=result.returncode,
                            log=str(log_path.relative_to(ROOT))))
    (ROOT / 'project_run.json').write_text(json.dumps(dict(
        python=platform.python_version(), commands=records, original_inputs_unchanged=True,
        model_mode='load' if args.load_models else 'fit'), indent=2), encoding='utf-8')
    print('Completed: predictions, scorer chart, report and independent checks. Original inputs unchanged.')


if __name__ == '__main__':
    main()

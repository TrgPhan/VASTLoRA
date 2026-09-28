"""Build a self-contained reviewed snapshot on top of a pinned public checkout."""

import base64
import hashlib
import json
from pathlib import Path
import subprocess
import zlib

ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "bbe40e0b4f91dd2be30e3a7dc273f1e1efb8ea18"
DESTINATION = ROOT / "notebooks/kaggle_qwen_1_5b_rift_causal_confirmation.ipynb"


def build():
    tracked = {
        line
        for line in subprocess.check_output(
            ["git", "ls-files"], cwd=ROOT, text=True
        ).splitlines()
        if line
    }
    paths = sorted(
        p
        for p in {
            *(ROOT / "src").rglob("*.py"),
            *(ROOT / "scripts").glob("*.py"),
            *(ROOT / "tests").glob("*.py"),
            *(ROOT / "configs").glob("*.json"),
            ROOT / "configs/week8_spectral_requirements.txt",
            ROOT / "pyproject.toml",
            ROOT / "VAST_LoRA_Research_Guide_12_Weeks.md",
        }
        if p.relative_to(ROOT).as_posix() in tracked
    )
    files = {
        p.relative_to(ROOT).as_posix(): p.read_text(encoding="utf-8") for p in paths
    }
    payload = json.dumps(files, sort_keys=True).encode()
    packed = base64.b64encode(zlib.compress(payload, 9)).decode()
    sha = hashlib.sha256(payload).hexdigest()
    cells = []

    def cell(kind, text, hidden=False):
        item = {
            "cell_type": kind,
            "id": f"causal-{len(cells)}",
            "metadata": {},
            "source": text.splitlines(keepends=True),
        }
        if kind == "code":
            item.update(execution_count=None, outputs=[])
        if hidden:
            item["metadata"] = {"jupyter": {"source_hidden": True}}
        cells.append(item)

    cell(
        "markdown",
        """# RIFT-Core: causal controls, Qwen2.5-1.5B, T4 x2

Enable Internet and GPU T4 x2. This notebook clones a pinned public GitHub
revision, then applies a SHA256-verified embedded snapshot of the reviewed code.
The runtime implementation is published at the pinned revision; the embedded
tracked-file snapshot also preserves the exact notebook release. No token or
private credential is included. Source/config hashes identify the actual run.

Modes: `preflight` (no training), `smoke` (QNLI, tiny budget), `development`,
`confirmation` (4 views, seeds 8201-8206). This is a NEW prospective study,
not the historical Week8 6101-6106 table. Content-group splits v2 fix duplicate
leakage and cannot be pooled with legacy results. Fresh seeds do not certify
untouched held-out data: confirmation requires your actual audit note.

Attribution has Core, Diag, no-repair, ordinary server A/B, server-only, raw.
Server controls are custom controls, not full-paper competitor reproductions.
Same labeled-data/step/gate budget does not mean same FLOPs. No automatic GO verdict.
""",
    )
    cell(
        "code",
        """from pathlib import Path
import base64, csv, hashlib, json, os, shutil, signal, subprocess, sys, time, zlib

RUN_MODE = 'preflight'  # preflight / smoke / development / confirmation
SUITE = 'attribution'  # attribution / delay / objective / tuning (development only)
RUN_TRAINING = False
TASKS = []  # [] = all four views; smoke defaults to ['qnli']
VARIANTS = []  # [] = all variants of the selected suite
SEEDS = None  # None = spec defaults; e.g. [8101] for a pilot or [8201, 8202, 8203]
GPU_IDS = [0, 1]  # One process/model per T4; VRAM is not pooled.
HELDOUT_AUDIT_NOTE = ''  # Required for confirmation; record actual history/split audit.
RESUME_ROOTS = [Path('/kaggle/input')]
REQUIRE_RESUME = False  # True blocks a fresh run when no matching artifacts were imported.
REQUIRE_SMOKE = False  # Optional safety gate; True requires matching QNLI smoke artifacts.
INSTALL_DEPENDENCIES = True
WORK_ROOT = Path('/kaggle/working')
REPO_URL = 'https://github.com/TrgPhan/VASTLoRA.git'
if RUN_MODE not in {'preflight', 'smoke', 'development', 'confirmation'}:
    raise ValueError('Unknown RUN_MODE')
if RUN_MODE == 'preflight' and RUN_TRAINING:
    raise ValueError('preflight cannot launch training')
if RUN_MODE == 'confirmation' and not HELDOUT_AUDIT_NOTE.strip():
    raise ValueError('Complete the held-out audit before confirmation; a new seed is not enough.')
WORK_ROOT.mkdir(parents=True, exist_ok=True)
""",
    )
    cell(
        "code",
        f"REPO_REF = {BASE_COMMIT!r}\nRELEASE_SHA256 = {sha!r}\nRELEASE_B64 = {packed!r}\n",
        hidden=True,
    )
    cell(
        "code",
        """payload = zlib.decompress(base64.b64decode(RELEASE_B64))
if hashlib.sha256(payload).hexdigest() != RELEASE_SHA256:
    raise RuntimeError('Corrupt source bundle')
release_files = json.loads(payload)
REPO_DIR = WORK_ROOT / ('RIFTLoRA-causal-' + RELEASE_SHA256[:12])
marker = REPO_DIR / '.causal_release.json'
if not REPO_DIR.exists():
    subprocess.run(['git', 'clone', '--no-checkout', REPO_URL, str(REPO_DIR)], check=True)
    subprocess.run(['git', 'fetch', 'origin', REPO_REF], cwd=REPO_DIR, check=True)
    subprocess.run(['git', 'checkout', '--detach', 'FETCH_HEAD'], cwd=REPO_DIR, check=True)
    for name, text in release_files.items():
        target = REPO_DIR / name
        if not target.resolve().is_relative_to(REPO_DIR.resolve()):
            raise ValueError('Unsafe bundle path')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding='utf-8')
    marker.write_text(json.dumps({'bundle_sha256': RELEASE_SHA256, 'base_commit': REPO_REF}), encoding='utf-8')
elif not marker.exists():
    raise RuntimeError('Unmarked/partial source directory. Inspect it and use a fresh WORK_ROOT; no overwrite.')
elif json.loads(marker.read_text())['bundle_sha256'] != RELEASE_SHA256:
    raise RuntimeError('Existing checkout belongs to another snapshot')
for name, text in release_files.items():
    if (REPO_DIR / name).read_text(encoding='utf-8') != text:
        raise RuntimeError('Source differs from bundled release: ' + name)
print('Verified source snapshot:', RELEASE_SHA256, '; base GitHub commit:', REPO_REF)

if INSTALL_DEPENDENCIES:
    import importlib.metadata
    constraint = WORK_ROOT / 'causal_torch_constraint.txt'
    constraint.write_text('torch==' + importlib.metadata.version('torch') + '\\n', encoding='utf-8')
    # Optional old torchao can make PEFT fail before training. Keep Kaggle's torch/CUDA wheel.
    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-c', str(constraint),
                    '-r', str(REPO_DIR / 'configs/week8_spectral_requirements.txt')], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-c', str(constraint),
                    '-e', '.[scale,dev]'], cwd=REPO_DIR, check=True)
sys.path.insert(0, str(REPO_DIR / 'src'))
sys.path.insert(0, str(REPO_DIR / 'scripts'))
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import riftlora
if not Path(riftlora.__file__).resolve().is_relative_to(REPO_DIR.resolve()):
    raise RuntimeError('Kernel imported a different checkout. Restart kernel.')
subprocess.run([sys.executable, '-c', 'import torch,peft,transformers,bitsandbytes; print(torch.__version__, peft.__version__, transformers.__version__, bitsandbytes.__version__); print("CUDA devices:", torch.cuda.device_count())'], check=True)
subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_rift_causal_study.py',
                'tests/test_core_repair.py', 'tests/test_scale_objective.py',
                'tests/test_rift_theory_contracts.py'], cwd=REPO_DIR, check=True)
""",
    )
    cell(
        "code",
        """from run_rift_causal_study import freeze, validate_manifest, completed_matches
from run_kaggle_3b import _config_fingerprint
from riftlora.scale.causal_study import fingerprint
phase = 'development' if RUN_MODE == 'preflight' else RUN_MODE
tasks = TASKS or (['qnli'] if phase == 'smoke' else None)
spec = json.loads((REPO_DIR / 'configs/rift_core_causal_study.json').read_text())
if SEEDS is not None:
    if not SEEDS or len(SEEDS) != len(set(SEEDS)):
        raise ValueError('SEEDS must be a non-empty list of unique integers')
    if phase not in spec['phases']:
        raise ValueError(f'No seed phase in spec: {phase}')
    spec['phases'][phase]['seeds'] = [int(seed) for seed in SEEDS]
plan = freeze(spec, suite=SUITE, phase=phase, tasks=tasks, variants=VARIANTS or None,
              attestation=HELDOUT_AUDIT_NOTE.strip() or None)
OUTPUT_ROOT = WORK_ROOT / ('rift_causal_v2_' + plan['manifest_sha256'][:12])
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
MANIFEST = OUTPUT_ROOT / 'manifest.json'
if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != plan:
    raise RuntimeError('Existing manifest differs')
MANIFEST.write_text(json.dumps(plan, indent=2), encoding='utf-8')
(OUTPUT_ROOT / 'source_snapshot.zlib').write_bytes(zlib.compress(payload, 9))
(OUTPUT_ROOT / 'release.json').write_text(json.dumps({'base_commit': REPO_REF,
    'bundle_sha256': RELEASE_SHA256, 'manifest_sha256': plan['manifest_sha256']}, indent=2))
with (OUTPUT_ROOT / 'pip_freeze.txt').open('w') as f:
    subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=f, check=True)
print({'phase': phase, 'suite': SUITE, 'jobs': len(plan['jobs']),
       'tasks': sorted({j['task'] for j in plan['jobs']}),
       'variants': sorted({j['variant'] for j in plan['jobs']}),
       'seeds': sorted({j['seed'] for j in plan['jobs']})})

# Data-only integration preflight: real cached GLUE, stop before model load.
from datasets import load_dataset
for subset in ['sst2', 'qnli', 'mnli']:
    base = plan['jobs'][0]['config']['dataset']
    load_dataset(base['hub_path'], subset, revision=base['revision'])
preflight_env = os.environ.copy()
preflight_env.update(RIFT_REAL_DATA_PREFLIGHT='1', CUDA_VISIBLE_DEVICES='')
subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_rift_causal_data_preflight.py'],
               cwd=REPO_DIR, env=preflight_env, check=True)
DATA_PREFLIGHT_OK = True
""",
    )
    cell(
        "code",
        """def job_path(job):
    return (OUTPUT_ROOT / phase / SUITE / job['task'] / job['regime'] / job['variant']
            / f"{job['variant']}_seed{job['seed']}" / 'result.json')

def valid_result(job, path):
    if not path.exists() or not completed_matches(path, job, job['config']):
        return False
    e, d = job['config']['experiment'], job['config']['dataset']
    expected = {'events.csv': e['warmup_returns'] + e['collected_returns'],
                'baseline_eval_details.csv': d['eval_examples'], 'final_eval_details.csv': d['eval_examples']}
    for name, count in expected.items():
        try:
            with (path.parent / name).open(encoding='utf-8', newline='') as handle:
                if sum(1 for _ in csv.DictReader(handle)) != count:
                    return False
        except (OSError, csv.Error):
            return False
    return True

lookup = {(_config_fingerprint(j['config']), j['variant'], j['seed']): j for j in plan['jobs']}
for source_root in RESUME_ROOTS:
    if not source_root.exists():
        continue
    for source in source_root.rglob('result.json'):
        try:
            r = json.loads(source.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            continue
        job = lookup.get((r.get('config_fingerprint'), r.get('variant'), r.get('seed')))
        if job is None or not valid_result(job, source):
            continue
        target = job_path(job)
        if target.exists():
            if not valid_result(job, target):
                raise RuntimeError('Conflicting local artifact: ' + str(target))
            continue
        if (target.parent / 'running.lock').exists():
            raise RuntimeError('Job lock exists; verify running processes before recovery')
        target.parent.mkdir(parents=True, exist_ok=True)
        for name in ('events.csv', 'baseline_eval_details.csv', 'final_eval_details.csv', 'result.json'):
            shutil.copy2(source.parent / name, target.parent / name)
completed = []
for job in plan['jobs']:
    path = job_path(job)
    if path.exists():
        if not valid_result(job, path):
            raise RuntimeError('Incomplete/mismatched artifact, refusing automatic overwrite: ' + str(path))
        completed.append(job)
if REQUIRE_RESUME and not completed:
    raise RuntimeError('No matching completed results imported; check attached Kaggle outputs')
print('Completed:', len(completed), 'Pending:', len(plan['jobs']) - len(completed))
RUNNER = REPO_DIR / 'scripts/run_rift_causal_study.py'
subprocess.run([sys.executable, str(RUNNER), '--execute', str(MANIFEST),
                '--output-root', str(OUTPUT_ROOT), '--dry-run'], cwd=REPO_DIR, check=True)
""",
    )
    cell(
        "code",
        """if RUN_TRAINING and len(completed) < len(plan['jobs']):
    if not DATA_PREFLIGHT_OK:
        raise RuntimeError('Data preflight is required')
    if REQUIRE_SMOKE and phase != 'smoke':
        smoke_plan = freeze(spec, suite=SUITE, phase='smoke', tasks=['qnli'], variants=VARIANTS or None)
        smoke_found = set()
        smoke_lookup = {(_config_fingerprint(j['config']), j['variant'], j['seed']): j for j in smoke_plan['jobs']}
        for root in [WORK_ROOT, *RESUME_ROOTS]:
            if not root.exists():
                continue
            for path in root.rglob('result.json'):
                try:
                    r = json.loads(path.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                key = (r.get('config_fingerprint'), r.get('variant'), r.get('seed'))
                smoke_job = smoke_lookup.get(key)
                if smoke_job is not None and valid_result(smoke_job, path):
                    smoke_found.add(key)
        if len(smoke_found) != len(smoke_plan['jobs']):
            raise RuntimeError('REQUIRE_SMOKE=True but matching QNLI smoke artifacts were not found.')
    import torch
    from huggingface_hub import snapshot_download
    if not GPU_IDS or len(GPU_IDS) != len(set(GPU_IDS)) or any(g < 0 or g >= torch.cuda.device_count() for g in GPU_IDS):
        raise RuntimeError('Select available distinct GPU IDs')
    model_spec = plan['jobs'][0]['config']['model']
    snapshot_download(model_spec['name'], revision=model_spec['revision'])
    validate_manifest(plan)
    workers = []
    try:
        for shard, gpu in enumerate(GPU_IDS):
            log_path = OUTPUT_ROOT / f'worker_gpu{gpu}.log'
            log = log_path.open('a', encoding='utf-8')
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1',
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
            command = [sys.executable, '-u', str(RUNNER), '--execute', str(MANIFEST),
                       '--output-root', str(OUTPUT_ROOT), '--shard-count', str(len(GPU_IDS)),
                       '--shard-index', str(shard)]
            try:
                process = subprocess.Popen(command, cwd=REPO_DIR, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
            except BaseException:
                log.close()
                raise
            workers.append((process, log, log_path))
        while any(p.poll() is None for p, _, _ in workers):
            if any(p.poll() not in (None, 0) for p, _, _ in workers):
                raise RuntimeError('Worker failed/OOM. Both queues stopped; inspect per-job run.log.')
            count = sum(job_path(j).exists() for j in plan['jobs'])
            print(time.strftime('%H:%M:%S'), 'result files:', count, '/', len(plan['jobs']), flush=True)
            time.sleep(30)
        if any(p.returncode != 0 for p, _, _ in workers):
            raise RuntimeError('Worker failed; inspect worker logs and per-job run.log')
    finally:
        for process, log, _ in workers:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            log.close()
    print('Workers completed.')
else:
    print('No training launched (preflight/disabled or all jobs completed).')
""",
    )
    cell(
        "code",
        """# Run this cell also after a stopped/failed run to export partial evidence.
from analyze_rift_causal_study import analyze, markdown
from IPython.display import Markdown, display, FileLink
report = analyze(plan, OUTPUT_ROOT)
(OUTPUT_ROOT / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
(OUTPUT_ROOT / 'report.md').write_text(markdown(report), encoding='utf-8')
display(Markdown(markdown(report)))
archive = shutil.make_archive(str(OUTPUT_ROOT) + '_results', 'zip', root_dir=OUTPUT_ROOT.parent, base_dir=OUTPUT_ROOT.name)
display(FileLink(archive))
print('Missing jobs:', report['missing'])
print('Partial/smoke results are not a confirmation verdict. No seed selection.')
""",
    )
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    DESTINATION.write_text(
        json.dumps(build(), ensure_ascii=True, indent=1) + "\n", encoding="utf-8"
    )
    print(DESTINATION)

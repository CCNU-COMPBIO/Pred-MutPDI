#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from common_DNA import inspect_complex, parse_mut, temporary_base

STRUCTURAL_NAMES = [
    'R-Value',
    'nucleotide/amino-acid interactions_w',
    'base-pair/amino-acid interactions_m',
    'Δ_base-pair/amino-acid interactions',
    'Δ_phosphate/amino-acid H-bonds',
    'Δ_base/amino-acid stacks',
]


def run_cmd(cmd: list[str], cwd: Path) -> str:
    result = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    if result.returncode:
        detail = (result.stdout + '\n' + result.stderr)[-12000:]
        if os.environ.get('PRED_MUTPDI_DEBUG') != '1':
            detail = 'Set PRED_MUTPDI_DEBUG=1 for the external-tool log.'
        raise RuntimeError(f'{Path(cmd[0]).name} exited with status {result.returncode}.\n{detail}')
    return result.stdout


def resolve_tool(root: Path, folder: str, names: tuple[str, ...], env_name: str) -> Path:
    override = os.environ.get(env_name)
    candidates = [Path(override).expanduser()] if override else [root / folder / n for n in names]
    if override and not Path(override).is_absolute() and len(Path(override).parts) == 1:
        found = shutil.which(override)
        if found:
            candidates.insert(0, Path(found))
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return path.resolve()
    raise FileNotFoundError(f'Executable missing: {folder}; set {env_name} or place it under {root / folder}.')


def find_foldx_exe(root: Path) -> Path:
    return resolve_tool(root, 'FoldX', ('FoldX', 'foldx', 'FoldX.exe', 'foldx.exe'), 'PRED_MUTPDI_FOLDX')


def find_dssr_exe(root: Path) -> Path:
    return resolve_tool(root, 'x3dna-dssr', ('x3dna-dssr', 'x3dna-dssr.exe'), 'PRED_MUTPDI_DSSR')


def parse_r_value(pdb_path: Path) -> float:
    candidates: list[tuple[int, float]] = []
    for line in Path(pdb_path).read_text(errors='replace').splitlines():
        if not re.match(r'^REMARK\s+3\s+', line):
            continue
        if 'FREE R' in line.upper() or ':' not in line:
            continue
        left, right = line.split(':', 1)
        upper = left.upper()
        if not re.search(r'\bR\s+VALUE\b', upper):
            continue
        # Ignore uncertainty/error and bin-specific statistics.
        if any(t in upper for t in ('ERROR', 'BIN', 'ESD')):
            continue
        try:
            value = float(right.strip().split()[0])
        except (ValueError, IndexError):
            continue
        if not math.isfinite(value) or not 0 <= value <= 1:
            continue
        priority = 0 if 'WORKING SET' in upper and 'TEST' not in upper else 1
        candidates.append((priority, value))
    if not candidates:
        raise ValueError('No unambiguous crystallographic R-value in PDB REMARK 3. '
                         'Use --r-value with the value/preprocessing used during DNA training; '
                         'a missing value is not silently replaced by resolution or zero.')
    return min(candidates, key=lambda x: x[0])[1]


def run_foldx_buildmodel(resource_dir: Path, pdb_id: str, chain: str, mut: str,
                         wt_pdb: Path, work_dir: Path) -> Path:
    exe = find_foldx_exe(resource_dir)
    wt, pos, mt = parse_mut(mut)
    job = work_dir / 'foldx_DNA'
    job.mkdir(parents=True, exist_ok=False)
    # A fresh workspace prevents an old <PDB>_1.pdb being reused or overwritten.
    local_input = job / f'{pdb_id}.pdb'
    shutil.copy2(wt_pdb, local_input)
    resource_dirs = {exe.parent, resource_dir / 'FoldX'}
    for directory in resource_dirs:
        for name in ('rotabase.txt', 'molecules', 'parameters'):
            source, dest = directory / name, job / name
            if source.exists() and not dest.exists():
                if source.is_dir():
                    dest.symlink_to(source.resolve(), target_is_directory=True)
                else:
                    shutil.copy2(source, dest)
    mutation_file = job / 'individual_list.txt'
    mutation_file.write_text(f'{wt}{chain}{pos}{mt};\n', encoding='utf-8')
    run_cmd([str(exe), '--command=BuildModel', f'--pdb={local_input.name}',
             f'--mutant-file={mutation_file.name}'], job)
    expected = job / f'{pdb_id}_1.pdb'
    if not expected.is_file():
        alternatives = [p for p in job.glob('*.pdb')
                        if p != local_input and not p.name.upper().startswith('WT')]
        if len(alternatives) != 1:
            raise RuntimeError(f'FoldX produced no unique mutant structure in {job}.')
        expected = alternatives[0]
    return expected


HEADINGS = {
    'nucleotide': re.compile(r'^\s*List\s+of\s+(\d+)\s+nucleotide/amino-acid interactions\b', re.I | re.M),
    'base_pair': re.compile(r'^\s*List\s+of\s+(\d+)\s+base-pair/amino-acid interactions\b', re.I | re.M),
    'phosphate_hb': re.compile(r'^\s*List\s+of\s+(\d+)\s+phosphate/amino-acid H-bonds\b', re.I | re.M),
    'base_stack': re.compile(r'^\s*List\s+of\s+(\d+)\s+base/amino-acid stacks\b', re.I | re.M),
}


def parse_snap_counts(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(f'SNAP report was not created: {path}')
    text = path.read_text(errors='replace')
    if not text.strip():
        raise ValueError(f'Empty SNAP report: {path}')
    # Validate the report separately from individual absent (zero-count) headings.
    if not re.search(r'(SNAP|DSSR|3DNA)', text, re.I):
        raise ValueError('Unrecognized SNAP report; review the DSSR version/output format.')
    result = {}
    for name, pattern in HEADINGS.items():
        hits = pattern.findall(text)
        if len(hits) > 1 and len(set(hits)) > 1:
            raise ValueError(f'Ambiguous duplicate {name} headings in SNAP report.')
        # As in the RNA source, a section omitted from a valid report means zero.
        result[name] = float(hits[0]) if hits else 0.0
    return result


def structural_from_counts(r_value: float, wild: dict, mutant: dict) -> dict[str, float]:
    return {
        'R-Value': float(r_value),
        'nucleotide/amino-acid interactions_w': wild['nucleotide'],
        'base-pair/amino-acid interactions_m': mutant['base_pair'],
        'Δ_base-pair/amino-acid interactions': mutant['base_pair'] - wild['base_pair'],
        'Δ_phosphate/amino-acid H-bonds': mutant['phosphate_hb'] - wild['phosphate_hb'],
        'Δ_base/amino-acid stacks': mutant['base_stack'] - wild['base_stack'],
    }


def compute_structural_features(pdb_id: str, chain: str, mut: str, *, wt_pdb: Path,
                                work_dir: Path, resource_dir: Path, r_value: float | None = None,
                                mutant_pdb: Path | None = None) -> tuple[dict[str, float], Path]:
    inspect_complex(wt_pdb, chain, mut)
    r = parse_r_value(wt_pdb) if r_value is None else float(r_value)
    if not math.isfinite(r) or not 0 <= r <= 1:
        raise ValueError('R-Value must be a finite value in [0, 1].')
    mutant = Path(mutant_pdb).resolve() if mutant_pdb else run_foldx_buildmodel(
        resource_dir, pdb_id, chain, mut, wt_pdb, work_dir)
    wt, pos, mt = parse_mut(mut)
    # Validate the supplied/generated mutant, including retained DNA chains.
    inspect_complex(mutant, chain, f'{mt}{pos}{wt}')
    exe = find_dssr_exe(resource_dir)
    reports = {}
    for tag, pdb in [('wt', wt_pdb), ('mt', mutant)]:
        folder = work_dir / f'snap_{tag}_DNA'
        folder.mkdir(parents=True, exist_ok=False)
        report = folder / 'report.txt'
        run_cmd([str(exe), 'snap', f'-i={pdb}', f'-o={report}'], folder)
        reports[tag] = parse_snap_counts(report)
    return structural_from_counts(r, reports['wt'], reports['mt']), mutant


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.parse_args()
    p.error('Internal feature component. Use predict_DNA.sh; use validate_features_DNA.py for auditing.')

if __name__ == '__main__':
    main()

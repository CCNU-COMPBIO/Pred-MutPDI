#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import contextlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
from common_DNA import (canonical_feature_name, inspect_complex, load_selection_order,
                        temporary_base, validate_identifiers)

BASE = Path(__file__).resolve().parent


def resolve_feature_order(booster, explicit_order: Path | None = None) -> tuple[list[str], list[str] | None]:
    expected = load_selection_order()
    expected_keys = {canonical_feature_name(n) for n in expected}
    n_features = int(booster.num_features())
    if n_features != 44:
        raise ValueError(f'This DNA adapter needs 44 model features; supplied model has {n_features}. '
                         'Do not use the 38-feature RNA model.')
    model_names = list(booster.feature_names) if booster.feature_names else None
    generic = model_names is None or model_names == [f'f{i}' for i in range(n_features)]
    supplied = load_selection_order(explicit_order) if explicit_order else None
    if not generic:
        order = model_names
        if supplied and [canonical_feature_name(n) for n in supplied] != [canonical_feature_name(n) for n in order]:
            raise ValueError('Explicit feature-order file conflicts with named feature order in the model.')
    elif supplied:
        order = supplied
    else:
        raise ValueError('Model has no biological feature names. Provide --feature-order pointing to the '
                         'EXACT training column order. forward_selection_history order is not automatically '
                         'assumed to be the model input order.')
    keys = [canonical_feature_name(n) for n in order]
    if len(set(keys)) != 44 or set(keys) != expected_keys:
        missing = sorted(expected_keys - set(keys))
        extra = sorted(set(keys)-expected_keys)
        raise ValueError(f'Model feature set differs from the supplied DNA selection. Missing={missing}; extra={extra}')
    return order, model_names


def load_model(path: Path):
    import xgboost as xgb
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'DNA model file not found: {path}')
    if path.suffix.lower() not in {'.json','.ubj'}:
        raise ValueError('Supply an XGBoost save_model .json or .ubj file, not a dump or pickle.')
    booster = xgb.Booster()
    booster.load_model(str(path))
    booster.set_param({'device': 'cpu'})
    return booster


def quiet_call(func, *args, **kwargs):
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            result = func(*args, **kwargs)
    except Exception as exc:
        details = (out.getvalue() + '\n' + err.getvalue()).strip()
        if os.environ.get('PRED_MUTPDI_DEBUG') == '1' and details:
            raise RuntimeError(f'{exc}\n{details}') from exc
        raise
    # Warnings must remain visible even though feature values stay hidden.
    if err.getvalue().strip():
        print(err.getvalue().strip(), file=sys.stderr)
    return result


def run_af3_es_via_pipe(pdb: str, chain: str, mut: str, wt_pdb: Path,
                       resource_dir: Path, allow_missing: bool = False, bash_exe: str = 'bash') -> float:
    if os.name != 'posix':
        raise RuntimeError('Full prediction uses Bash and inherited file descriptors. Use Linux or WSL2.')
    read_fd, write_fd = os.pipe()
    env = os.environ.copy()
    env['PRED_MUTPDI_RESULT_FD'] = str(write_fd)
    command = [bash_exe, str(BASE/'run_af3_all_DNA.sh'), '-PDB', pdb, '-CHAIN', chain,
               '-Mut', mut, '--pdb-path', str(wt_pdb), '--resource-dir', str(resource_dir)]
    if allow_missing:
        command.append('--allow-missing-es')
    try:
        proc = subprocess.Popen(command, env=env, pass_fds=(write_fd,),
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        os.close(write_fd)
        write_fd = -1
        _, error = proc.communicate()
        if proc.returncode:
            raise RuntimeError(f'AF3 stage failed (exit {proc.returncode}).\n{error.strip()}')
        data = []
        while True:
            block = os.read(read_fd, 4096)
            if not block:
                break
            data.append(block)
        payload = b''.join(data).decode('ascii').strip()
        if not payload or len(payload.splitlines()) != 1:
            raise ValueError('AF3 result channel did not return exactly one scalar.')
        value = float(payload)
        if math.isinf(value) or (math.isnan(value) and not allow_missing):
            raise ValueError('Invalid ES result from AF3.')
        if error.strip():
            print(error.strip(), file=sys.stderr)
        return value
    finally:
        os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)


def compute_feature_map(args, stage: str = 'all') -> dict[str, float]:
    """Shared extraction path; the explicit audit tool can request one stage."""
    validate_identifiers(args.PDB, args.CHAIN, args.Mut)
    wt_pdb = (args.pdb_path or args.input_dir/f'{args.PDB}.pdb').expanduser().resolve()
    resource_dir = args.resource_dir.expanduser().resolve()
    inspect_complex(wt_pdb, args.CHAIN, args.Mut)
    selected_stages = {'structural','network','es','esm'} if stage == 'all' else {stage}
    features = {}
    with tempfile.TemporaryDirectory(prefix='predmutpdi_DNA_', dir=temporary_base()) as temp:
        work = Path(temp)
        work.chmod(0o700)
        mutant = args.mutant_pdb.expanduser().resolve() if args.mutant_pdb else None
        if 'structural' in selected_stages:
            from structural_feature_DNA import compute_structural_features
            structural, mutant = quiet_call(compute_structural_features, args.PDB, args.CHAIN, args.Mut,
                                             wt_pdb=wt_pdb, work_dir=work, resource_dir=resource_dir,
                                             r_value=args.r_value, mutant_pdb=mutant)
            features.update(structural)
        if 'network' in selected_stages:
            if mutant is None:
                from structural_feature_DNA import run_foldx_buildmodel
                mutant = quiet_call(run_foldx_buildmodel, resource_dir, args.PDB, args.CHAIN,
                                    args.Mut, wt_pdb, work)
            from run_network_DNA import compute_network_features
            # Ensure a precomputed mutant actually contains the requested substitution.
            wt, pos, mt = validate_identifiers(args.PDB, args.CHAIN, args.Mut)
            inspect_complex(mutant, args.CHAIN, f'{mt}{pos}{wt}')
            features.update(quiet_call(compute_network_features, args.PDB, args.CHAIN, args.Mut,
                                      wt_pdb=wt_pdb, mut_pdb=mutant))
        if 'es' in selected_stages:
            if args.af3_root:
                from calculating_es_DNA import compute_avg_es
                es = quiet_call(compute_avg_es, args.PDB, args.CHAIN, args.Mut, wt_pdb=wt_pdb,
                                af3_root=args.af3_root, mapping_path=args.af3_mapping,
                                allow_missing=args.allow_missing_es)
                # Match the inherited-pipe six-decimal precision of the RNA source.
                es = float(f'{es:.6f}')
            else:
                es = run_af3_es_via_pipe(args.PDB, args.CHAIN, args.Mut, wt_pdb,
                                        resource_dir, args.allow_missing_es, args.bash)
            features['Avg_ES'] = es
        if 'esm' in selected_stages:
            from run_esm_DNA import compute_esm_feature
            features['Hi'] = quiet_call(compute_esm_feature, args.PDB, args.CHAIN, args.Mut, wt_pdb=wt_pdb)
    return features


def assemble_vector(features: dict[str, float], order: list[str], allow_missing_es: bool = False) -> np.ndarray:
    canonical = {}
    for key, value in features.items():
        ck = canonical_feature_name(key)
        if ck in canonical:
            raise ValueError(f'Duplicate normalized feature name: {key}')
        canonical[ck] = value
    ordered_keys = [canonical_feature_name(n) for n in order]
    if set(canonical) != set(ordered_keys):
        raise ValueError('Extracted feature keys do not match the model feature set.')
    array = np.asarray([canonical[k] for k in ordered_keys], dtype=float)
    for name, value in zip(order, array):
        if math.isnan(value) and name == 'Avg_ES' and allow_missing_es:
            continue
        if not math.isfinite(value):
            raise ValueError(f'Non-finite feature: {name}')
    return array


def add_input_arguments(parser):
    parser.add_argument('-PDB')
    parser.add_argument('-CHAIN')
    parser.add_argument('-Mut')
    parser.add_argument('--input-dir', type=Path, default=BASE,
                        help='Directory containing <PDB>.pdb; default is the script directory.')
    parser.add_argument('--pdb-path', type=Path, help='Explicit input PDB path.')
    parser.add_argument('--resource-dir', type=Path, default=BASE,
                        help='Directory containing FoldX/, x3dna-dssr/, alphafold3/.')
    parser.add_argument('--mutant-pdb', type=Path, help='Optional already generated FoldX mutant with original PDB numbering.')
    parser.add_argument('--r-value', type=float, help='Explicit R-Value when REMARK 3 is absent; must match training preprocessing.')
    parser.add_argument('--af3-root', type=Path, help='Existing per-mutation AF3 sample directory; skips a new AF3 run.')
    parser.add_argument('--af3-mapping', type=Path, help='Mapping JSON from run_af3_mutation_DNA.py for existing AF3 output.')
    parser.add_argument('--allow-missing-es', action='store_true',
                        help='Allow NaN Avg_ES only if all located samples fail; no other feature may be nonfinite.')
    parser.add_argument('--bash', default='bash')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_input_arguments(p)
    p.add_argument('--model_json', '--model-json', type=Path, default=BASE/'model_DNA.json',
                   help='XGBoost model; default is bundled model_DNA.json.')
    p.add_argument('--feature-order', type=Path,
                   help='Required for unnamed models: exact training order, NOT an assumed ordering.')
    p.add_argument('--check-model', action='store_true', help='Check model width and feature names without extracting features.')
    p.add_argument('--iteration-limit', type=int, help='Optional exclusive upper tree-iteration bound; default uses all saved trees.')
    a = p.parse_args()
    booster = load_model(a.model_json)
    order, model_names = resolve_feature_order(booster, a.feature_order)
    if a.check_model:
        print(json.dumps({'model': str(a.model_json.resolve()), 'num_features': booster.num_features(),
                          'model_feature_names_present': bool(model_names), 'resolved_feature_order': order,
                          'best_iteration_attribute': booster.attr('best_iteration'),
                          'note': 'Feature identities/order checked; numerical extraction is not validated by this check.'},
                         ensure_ascii=False, indent=2))
        return
    if not (a.PDB and a.CHAIN and a.Mut):
        p.error('-PDB, -CHAIN and -Mut are required for prediction.')
    if booster.attr('best_iteration') is not None and a.iteration_limit is None:
        print('[WARNING] Model contains best_iteration. This adapter uses all saved trees, like the RNA source; '
              'supply --iteration-limit only if the training prediction used that limit.', file=sys.stderr)
    feature_map = compute_feature_map(a)
    vector = assemble_vector(feature_map, order, a.allow_missing_es)
    import xgboost as xgb
    matrix = xgb.DMatrix(vector.reshape(1,-1), feature_names=model_names)
    kwargs = {'iteration_range': (0,a.iteration_limit)} if a.iteration_limit else {}
    if a.iteration_limit is not None and a.iteration_limit <= 0:
        raise ValueError('--iteration-limit must be positive.')
    try:
        prediction = float(np.asarray(booster.predict(matrix, **kwargs)).ravel()[0])
        if not math.isfinite(prediction):
            raise ValueError('Non-finite model prediction.')
    finally:
        vector.fill(0)
        feature_map.clear()
    print(f'{prediction:.6f}')

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if os.environ.get('PRED_MUTPDI_DEBUG') == '1':
            raise
        print(f'[ERROR] {exc}', file=sys.stderr)
        sys.exit(1)

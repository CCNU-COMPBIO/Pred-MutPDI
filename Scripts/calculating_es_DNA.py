#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
import numpy as np
from Bio import PDB
from common_DNA import AA3_TO_1, build_af3_input, parse_mut


def get_rep_atom(res):
    for name in ('CA', 'C', 'N'):
        if name in res:
            return res[name]
    return None


def kabsch_fit(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if P.shape != Q.shape or P.ndim != 2 or P.shape[1] != 3 or len(P) < 3:
        raise ValueError('Kabsch alignment requires at least three paired 3D atoms.')
    cp, cq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((Q-cq).T @ (P-cp))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R, cp - cq @ R


def _residue_map(chain) -> dict[int, object]:
    return {int(r.id[1]): r for r in chain
            if r.resname.strip().upper() in AA3_TO_1 and not r.id[2].strip()}


def paired_residues(struct_wt, struct_mut, mapping: dict) -> dict:
    wt_model, mt_model = next(struct_wt.get_models()), next(struct_mut.get_models())
    pairs = {}
    for c in mapping['chains']:
        if c['type'] != 'protein':
            continue
        wc, mc = c['original_chain'], c['af3_chain']
        if wc not in wt_model or mc not in mt_model:
            raise ValueError(f'Protein-chain mapping not found in structures: {wc} -> {mc}')
        wmap, mmap = _residue_map(wt_model[wc]), _residue_map(mt_model[mc])
        for number, afnumber, wtaa, mtaa in zip(c['residue_numbers'], c['af3_residue_numbers'],
                                               c['sequence'], c['mutant_sequence']):
            if number not in wmap or afnumber not in mmap:
                raise ValueError(f'Missing residue for mapping {wc}:{number} -> {mc}:{afnumber}.')
            wr, mr = wmap[number], mmap[afnumber]
            if AA3_TO_1[wr.resname.strip().upper()] != wtaa or AA3_TO_1[mr.resname.strip().upper()] != mtaa:
                raise ValueError(f'Sequence mismatch in WT/AF3 mapping at {wc}:{number}.')
            pairs[(wc, number)] = (wr, mr)
    return pairs


def compute_es_from_pairs(pairs: dict, mutated_chain: str, mutated_resi: int, cutoff: float = 13.0) -> float:
    key = (mutated_chain, mutated_resi)
    if key not in pairs:
        raise ValueError('Mutation site absent from mapped protein residues.')
    anchor_w, anchor_m = [], []
    for (chain, _), (wr, mr) in pairs.items():
        if chain == mutated_chain:
            for atom in ('N', 'CA', 'C'):
                if atom in wr and atom in mr:
                    anchor_w.append(wr[atom].coord)
                    anchor_m.append(mr[atom].coord)
    R, t = kabsch_fit(np.asarray(anchor_w, dtype=float), np.asarray(anchor_m, dtype=float))
    wr, mr = pairs[key]
    wa, ma = get_rep_atom(wr), get_rep_atom(mr)
    if wa is None or ma is None:
        raise ValueError('Mutation site lacks a representative backbone atom.')
    center_w = np.asarray(wa.coord, dtype=float)
    center_m = np.asarray(ma.coord, dtype=float) @ R + t
    ratios = []
    for other, (wr, mr) in pairs.items():
        if other == key:
            continue
        wa, ma = get_rep_atom(wr), get_rep_atom(mr)
        if wa is None or ma is None:
            continue
        v_w = np.asarray(wa.coord, dtype=float) - center_w
        distance = np.linalg.norm(v_w)
        if not 1e-6 <= distance < cutoff:
            continue
        v_m = np.asarray(ma.coord, dtype=float) @ R + t - center_m
        ratios.append(float(np.linalg.norm(v_w-v_m)/distance))
    if not ratios:
        raise ValueError('No valid protein neighbors for ES.')
    return float(np.mean(ratios))


def find_seed_cif_files(af3_root: Path, job_name: str, original_label: str) -> list[Path]:
    root = Path(af3_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'AF3 output directory does not exist: {root}')
    names = {job_name.lower(), original_label.lower()}
    direct_samples = list(root.glob('seed-1_sample-*'))
    if root.name.lower() in names or direct_samples:
        search_root = root
    else:
        candidates = [p for p in root.glob('*') if p.is_dir() and p.name.lower() in names]
        if not candidates:
            candidates = [p for p in root.glob('*/*') if p.is_dir() and p.name.lower() in names]
        if len(candidates) != 1:
            raise ValueError('Cannot identify one AF3 job directory for this mutation. '
                             'Pass the exact per-mutation --af3-root directory.')
        search_root = candidates[0]
    grouped = {}
    for path in search_root.rglob('*.cif'):
        sample_parts = [re.fullmatch(r'seed-1_sample-(\d+)', part) for part in path.parts]
        sample = next((int(m.group(1)) for m in sample_parts if m), None)
        if sample is not None:
            grouped.setdefault(sample, []).append(path)
    if not grouped:
        raise FileNotFoundError('No seed-1_sample-*/model.cif files found for this mutation.')
    result = []
    for index in sorted(grouped):
        paths = grouped[index]
        if len(paths) > 1:
            preferred = [p for p in paths if p.name == 'model.cif']
            if len(preferred) != 1:
                raise ValueError(f'Ambiguous CIF files for sample {index}: {paths}')
            paths = preferred
        result.append(paths[0])
    return result


def compute_avg_es(pdb_id: str, chain: str, mut: str, *, wt_pdb: Path,
                   af3_root: Path, mapping_path: Path | None = None, cutoff: float = 13.0,
                   allow_missing: bool = False) -> float:
    _, pos, _ = parse_mut(mut)
    if mapping_path:
        mapping = json.loads(Path(mapping_path).read_text(encoding='utf-8'))
        if mapping['pdb_id'] != pdb_id or mapping['mutation_chain'] != chain or mapping['mutation'] != mut.upper():
            raise ValueError('AF3 mapping belongs to a different mutation.')
    else:
        _, mapping = build_af3_input(pdb_id, chain, mut, wt_pdb)
    files = find_seed_cif_files(af3_root, mapping['job_name'], f'{pdb_id}_{chain}_{mut}')
    wt_struct = PDB.PDBParser(QUIET=True).get_structure('wt', str(wt_pdb))
    parser = PDB.MMCIFParser(QUIET=True)
    values, failed = [], []
    for path in files:
        try:
            mt_struct = parser.get_structure(path.stem, str(path))
            value = compute_es_from_pairs(paired_residues(wt_struct, mt_struct, mapping), chain, pos, cutoff)
            if not math.isfinite(value):
                raise ValueError('Non-finite sample ES.')
            values.append(value)
        except Exception as exc:
            failed.append(f'{path}: {exc}')
    if failed:
        message = f'[WARNING] {len(failed)}/{len(files)} AF3 samples failed ES validation.'
        if os.environ.get('PRED_MUTPDI_DEBUG') == '1':
            message += '\n' + '\n'.join(failed)
        print(message, file=sys.stderr)
    if not values:
        if allow_missing:
            return float('nan')
        raise ValueError('No valid ES samples; prediction stopped. Use --allow-missing-es only if '
                         'this missing-value policy matches the trained DNA model.')
    return float(np.mean(values))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-PDB', required=True)
    p.add_argument('-CHAIN', required=True)
    p.add_argument('-Mut', required=True)
    p.add_argument('--pdb-path', required=True, type=Path)
    p.add_argument('--af3-root', required=True, type=Path)
    p.add_argument('--mapping-json', type=Path)
    p.add_argument('--cutoff', type=float, default=13.0)
    p.add_argument('--allow-missing-es', action='store_true')
    a = p.parse_args()
    fd = os.environ.get('PRED_MUTPDI_RESULT_FD')
    if not fd:
        p.error('Internal result channel missing. Use predict_DNA.sh.')
    value = compute_avg_es(a.PDB, a.CHAIN, a.Mut, wt_pdb=a.pdb_path,
                           af3_root=a.af3_root, mapping_path=a.mapping_json,
                           cutoff=a.cutoff, allow_missing=a.allow_missing_es)
    # Same six-decimal transport precision as the supplied RNA pipeline.
    os.write(int(fd), f'{value:.6f}\n'.encode('ascii'))

if __name__ == '__main__':
    main()

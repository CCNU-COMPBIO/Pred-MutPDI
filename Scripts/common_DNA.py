#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

AA3_TO_1 = dict(zip(
    'ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR'.split(),
    'ACDEFGHIKLMNPQRSTVWY'))
AA20 = list('ACDEFGHIKLMNPQRSTVWY')
DNA_MAP = {'DA': 'A', 'DC': 'C', 'DG': 'G', 'DT': 'T',
           'A': 'A', 'C': 'C', 'G': 'G', 'T': 'T'}
RNA_NAMES = {'U', 'DU', 'RA', 'RC', 'RG', 'RU', 'RT'}
ALLOWED_MODIFIED_DNA = {'BRU'}
MUT_RE = re.compile(r'^([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$')


def parse_mut(mut: str) -> tuple[str, int, str]:
    match = MUT_RE.fullmatch(mut.strip().upper())
    if not match:
        raise ValueError(f'Invalid mutation {mut!r}; expected a single substitution such as R55A.')
    wt, pos, mt = match.groups()
    if wt == mt:
        raise ValueError('Wild-type and mutant amino acids are identical.')
    return wt, int(pos), mt


def validate_identifiers(pdb_id: str, chain: str, mut: str):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', pdb_id) or '..' in pdb_id:
        raise ValueError('PDB identifier must be a safe filename stem, without a path.')
    if len(chain) != 1 or not chain.isalnum():
        raise ValueError('This PDB workflow requires a one-character alphanumeric chain ID.')
    return parse_mut(mut)


def load_first_model(path: Path):
    from Bio.PDB import PDBParser
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'PDB file not found: {path}')
    structure = PDBParser(QUIET=True).get_structure(path.stem, str(path))
    return structure, next(structure.get_models())


def residue_kind(res) -> str:
    name = res.resname.strip().upper()
    if name in AA3_TO_1 and res.id[0] == ' ':
        return 'protein'
    if name in RNA_NAMES:
        return 'rna'
    if name in DNA_MAP:
        # One-letter A/C/G names are ambiguous; sugar chemistry disambiguates.
        if name in {'A', 'C', 'G', 'T'} and ("O2'" in res or 'O2*' in res):
            return 'rna'
        return 'dna'
    atoms = {a.name.strip() for a in res}
    if name in ALLOWED_MODIFIED_DNA:
        return 'modified_dna'
    if {'N', 'CA', 'C'}.issubset(atoms) or "C1'" in atoms or 'C1*' in atoms:
        return 'modified_polymer'
    return 'other'


def inspect_complex(path: Path, chain: str, mut: str) -> dict:
    """Validate first-model polymer chains; return exact observed-residue sequences.

    No SEQRES reconstruction, missing-loop modeling, or nucleotide replacement is
    performed. This follows the uploaded workflow's use of coordinate-derived
    protein sequences. AF3 sequence positions are recorded explicitly.
    """
    wt, pos, mt = parse_mut(mut)
    _, model = load_first_model(path)
    chains = []
    ignored = set()
    for ch in model:
        residues = []
        kinds = set()
        for res in ch:
            kind = residue_kind(res)
            if kind == 'other':
                if res.resname.strip() not in {'HOH', 'WAT'}:
                    ignored.add(res.resname.strip())
                continue
            if kind == 'modified_polymer':
                raise ValueError(f'Unsupported modified polymer residue {res.resname} in '
                                 f'{ch.id}:{res.id[1]}. Supply the same standardized structure used during training.')
            if kind == 'modified_dna':
                # Historical Pred-MutPDI handling: retain the original PDB for FoldX/SNAP,
                # while excluding unsupported modified bases from sequence-only AF3 input.
                ignored.add(f'{res.resname.strip()}@{ch.id}:{res.id[1]}')
                continue
            if kind == 'rna':
                raise ValueError(f'RNA residue detected in chain {ch.id}; this adapter is for protein-DNA inputs.')
            if res.id[2].strip():
                raise ValueError(f'Insertion code at {ch.id}:{res.id[1]}{res.id[2]} is not supported by this '
                                 'FoldX mutation interface. Use a documented, consistently renumbered input.')
            kinds.add(kind)
            residues.append(res)
        if not residues:
            continue
        if len(kinds) != 1:
            raise ValueError(f'Chain {ch.id} mixes polymer types.')
        kind = next(iter(kinds))
        mapping = AA3_TO_1 if kind == 'protein' else DNA_MAP
        sequence = ''.join(mapping[r.resname.strip().upper()] for r in residues)
        numbers = [int(r.id[1]) for r in residues]
        if len(numbers) != len(set(numbers)):
            raise ValueError(f'Duplicate residue numbers in chain {ch.id}.')
        chains.append({'original_chain': ch.id, 'type': kind, 'sequence': sequence,
                       'residue_numbers': numbers})
    if not any(c['type'] == 'dna' for c in chains):
        raise ValueError('No supported DNA chain was found in the first PDB model.')
    target = next((c for c in chains if c['original_chain'] == chain and c['type'] == 'protein'), None)
    if target is None:
        raise ValueError(f'Mutation chain {chain!r} is not a supported protein chain.')
    if pos not in target['residue_numbers']:
        raise ValueError(f'Residue {pos} is not present in protein chain {chain}.')
    idx = target['residue_numbers'].index(pos)
    if target['sequence'][idx] != wt:
        raise ValueError(f'WT mismatch at {chain}:{pos}: input has {target["sequence"][idx]}, mutation says {wt}.')
    return {'chains': chains, 'mutation_chain': chain, 'mutation': mut.upper(),
            'mutation_index0': idx, 'ignored_nonpolymer_residues': sorted(ignored)}


def _letter_id(number: int) -> str:
    text = ''
    while True:
        number, rem = divmod(number, 26)
        text = chr(65 + rem) + text
        if number == 0:
            return text
        number -= 1


def build_af3_input(pdb_id: str, chain: str, mut: str, pdb_path: Path) -> tuple[dict, dict]:
    validate_identifiers(pdb_id, chain, mut)
    info = inspect_complex(pdb_path, chain, mut)
    _, _, mt = parse_mut(mut)
    # Preserve valid uppercase IDs; remap numeric/lowercase chains without collisions.
    used = {c['original_chain'] for c in info['chains']
            if re.fullmatch('[A-Z]+', c['original_chain'])}
    count = 0
    entries, maps = [], []
    for c in info['chains']:
        original = c['original_chain']
        af_id = original
        if not re.fullmatch('[A-Z]+', original):
            while _letter_id(count) in used:
                count += 1
            af_id = _letter_id(count)
            used.add(af_id)
            count += 1
        seq = c['sequence']
        if original == chain:
            k = info['mutation_index0']
            seq = seq[:k] + mt + seq[k+1:]
        entries.append({c['type']: {'id': [af_id], 'sequence': seq}})
        maps.append({**c, 'af3_chain': af_id, 'mutant_sequence': seq,
                     'af3_residue_numbers': list(range(1, len(seq)+1))})
    job_name = re.sub(r'[^a-z0-9_-]', '_', f'{pdb_id}_{chain}_{mut}_DNA'.lower())
    payload = {'name': job_name, 'sequences': entries, 'modelSeeds': [1],
               'dialect': 'alphafold3', 'version': 1}
    metadata = {'schema_version': 1, 'pdb_id': pdb_id, 'mutation': mut.upper(),
                'mutation_chain': chain, 'job_name': job_name, 'chains': maps,
                'ignored_nonpolymer_residues': info['ignored_nonpolymer_residues']}
    return payload, metadata


def canonical_feature_name(name: str) -> str:
    """Normalize dash glyphs only, not AAinter/NAinter or feature semantics."""
    return str(name).strip().replace('–', '-').replace('−', '-').replace('—', '-')


def load_selection_order(path: Path | None = None) -> list[str]:
    path = path or Path(__file__).resolve().with_name('feature_order_DNA.json')
    obj = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    order = obj.get('feature_order', obj.get('features')) if isinstance(obj, dict) else obj
    if not isinstance(order, list) or not all(isinstance(n, str) and n.strip() for n in order):
        raise ValueError('Feature-order file must contain a list of feature names.')
    keys = [canonical_feature_name(n) for n in order]
    if len(keys) != 44 or len(set(keys)) != 44:
        raise ValueError('DNA feature order must contain 44 unique names.')
    return order


def temporary_base() -> str | None:
    value = os.environ.get('PRED_MUTPDI_TMPDIR') or None
    if value:
        Path(value).mkdir(parents=True, exist_ok=True)
    return value

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import json
from pathlib import Path
from common_DNA import build_af3_input


def generate_af3_json(pdb_id: str, chain_id: str, mut: str, pdb_path: Path,
                      json_path: Path, mapping_path: Path | None = None):
    payload, mapping = build_af3_input(pdb_id, chain_id, mut, Path(pdb_path))
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2)+'\n', encoding='utf-8')
    if mapping_path is not None:
        Path(mapping_path).write_text(json.dumps(mapping, indent=2)+'\n', encoding='utf-8')
    return mapping


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-PDB', required=True)
    p.add_argument('-CHAIN', required=True)
    p.add_argument('-Mut', required=True)
    p.add_argument('--pdb-path', type=Path)
    p.add_argument('--json_path', '--json-path', required=True, type=Path)
    p.add_argument('--mapping-json', type=Path)
    a = p.parse_args()
    source = a.pdb_path or Path(__file__).resolve().with_name(f'{a.PDB}.pdb')
    generate_af3_json(a.PDB, a.CHAIN, a.Mut, source, a.json_path, a.mapping_json)


if __name__ == '__main__':
    main()

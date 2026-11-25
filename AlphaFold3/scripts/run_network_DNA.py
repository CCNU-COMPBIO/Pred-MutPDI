#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import math
import re
import time
import uuid
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from common_DNA import AA3_TO_1, DNA_MAP, parse_mut, load_selection_order

AA3 = set(AA3_TO_1)
NT_NAMES = set(DNA_MAP)
PARSER = PDBParser(QUIET=True)
p4c = None
NODE_FEATURES = [
    'AverageShortestPathLength','ClosenessCentrality','Eccentricity','Stress',
    'Degree','BetweennessCentrality','NeighborhoodConnectivity',
    'NumberOfUndirectedEdges','Radiality','TopologicalCoefficient']
STRUCTURAL_NAMES = {
    'R-Value','nucleotide/amino-acid interactions_w',
    'base-pair/amino-acid interactions_m','Δ_base-pair/amino-acid interactions',
    'Δ_phosphate/amino-acid H-bonds','Δ_base/amino-acid stacks'}
FEATURE_ORDER = [n for n in load_selection_order()
                 if n not in STRUCTURAL_NAMES and n not in {'Hi','Avg_ES'}]


def _connect_cytoscape():
    global p4c
    if p4c is None:
        import py4cytoscape as api
        p4c = api
    p4c.cytoscape_ping()
    return p4c

def safe_filename(s: str) -> str:
    s = str(s)
    s = re.sub(r'[\\/:*?"<>|]+', "_", s).strip().strip(".")
    return s[:180] if len(s) > 180 else s

def get_structure(pdb_path: Path):
    return PARSER.get_structure(pdb_path.stem, str(pdb_path))

def is_standard_residue(res) -> bool:
    return (
        res.get_resname().strip().upper() in AA3
        and res.id[0] == " "
    )

def is_nucleotide(res) -> bool:
    return res.get_resname().strip().upper() in NT_NAMES

def residue_uid(chain_id: str, res) -> str:
    resname = res.get_resname().strip().upper()
    resseq = int(res.id[1])
    return f"{resname}_{chain_id}{resseq}"

def atoms_of_residue(res):
    # Preserve all atoms present in the PDB, including hydrogens.
    return list(res.get_atoms())

def find_residue(structure, chain_id: str, pos: int):
    model = next(structure.get_models())
    if chain_id not in model:
        raise ValueError(
            f"Chain {chain_id} not found in structure {structure.id}"
        )
    chain = model[chain_id]
    for res in chain.get_residues():
        if res.id[0] != " ":
            continue
        if int(res.id[1]) == pos:
            return chain, res
    raise ValueError(
        f"Residue position {pos} not found in chain {chain_id}"
    )

def add_source_target(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    df = df.copy()

    if kind == "AA":
        c1, c2, c3, c4 = (
            "Target_Residue",
            "Target_Atom",
            "Neighbor_Residue",
            "Neighbor_Atom",
        )
    elif kind == "NA":
        c1, c2, c3, c4 = (
            "Protein_Residue",
            "Protein_Atom",
            "Nucleic_Residue",
            "Nucleic_Atom",
        )
    else:
        raise ValueError("kind must be 'AA' or 'NA'")

    if df.empty:
        df["source"] = pd.Series(dtype=str)
        df["target"] = pd.Series(dtype=str)
    else:
        df["source"] = (
            df[c1].astype(str) + "_" + df[c2].astype(str)
        )
        df["target"] = (
            df[c3].astype(str) + "_" + df[c4].astype(str)
        )

    base_cols = [
        c for c in df.columns
        if c not in ("source", "target")
    ]
    return df[base_cols + ["source", "target"]]

def extract_aa_aa_contacts(
    pdb_path: Path,
    chain_id: str,
    mut_pos: int,
    cutoff: float = 5.0,
) -> pd.DataFrame:
    structure = get_structure(pdb_path)
    _, mut_res = find_residue(structure, chain_id, mut_pos)

    mut_atoms = atoms_of_residue(mut_res)
    columns = [
        "Target_Residue",
        "Target_Atom",
        "Neighbor_Residue",
        "Neighbor_Atom",
        "Distance",
    ]
    if not mut_atoms:
        return add_source_target(
            pd.DataFrame(columns=columns),
            kind="AA",
        )

    rows = []
    mut_uid = residue_uid(chain_id, mut_res)
    model = next(structure.get_models())

    for ch in model.get_chains():
        ch_id = ch.id
        for res in ch.get_residues():
            if not is_standard_residue(res):
                continue
            if is_nucleotide(res):
                continue
            if ch_id == chain_id and res.id == mut_res.id:
                continue

            neigh_atoms = atoms_of_residue(res)
            if not neigh_atoms:
                continue

            neigh_uid = residue_uid(ch_id, res)
            for a in mut_atoms:
                a_coord = a.coord
                for b in neigh_atoms:
                    d = float(np.linalg.norm(a_coord - b.coord))
                    if d <= cutoff:
                        rows.append({
                            "Target_Residue": mut_uid,
                            "Target_Atom": a.get_name().strip(),
                            "Neighbor_Residue": neigh_uid,
                            "Neighbor_Atom": b.get_name().strip(),
                            "Distance": d,
                        })

    df = pd.DataFrame(rows, columns=columns)
    return add_source_target(df, kind="AA")

def extract_aa_na_contacts(
    pdb_path: Path,
    chain_id: str,
    mut_pos: int,
    cutoff: float = 10.0,
) -> pd.DataFrame:
    structure = get_structure(pdb_path)
    _, mut_res = find_residue(structure, chain_id, mut_pos)

    mut_atoms = atoms_of_residue(mut_res)
    columns = [
        "Protein_Residue",
        "Protein_Atom",
        "Nucleic_Residue",
        "Nucleic_Atom",
        "Distance",
    ]
    if not mut_atoms:
        return add_source_target(
            pd.DataFrame(columns=columns),
            kind="NA",
        )

    rows = []
    prot_uid = residue_uid(chain_id, mut_res)
    model = next(structure.get_models())

    for ch in model.get_chains():
        ch_id = ch.id
        for res in ch.get_residues():
            if not is_nucleotide(res):
                continue

            nt_atoms = atoms_of_residue(res)
            if not nt_atoms:
                continue

            nt_uid = residue_uid(ch_id, res)
            for a in mut_atoms:
                a_coord = a.coord
                for b in nt_atoms:
                    d = float(np.linalg.norm(a_coord - b.coord))
                    if d <= cutoff:
                        rows.append({
                            "Protein_Residue": prot_uid,
                            "Protein_Atom": a.get_name().strip(),
                            "Nucleic_Residue": nt_uid,
                            "Nucleic_Atom": b.get_name().strip(),
                            "Distance": d,
                        })

    df = pd.DataFrame(rows, columns=columns)
    return add_source_target(df, kind="NA")

def edge_frame_from_interactions(
    interaction_df: pd.DataFrame,
) -> pd.DataFrame:
    if interaction_df is None or interaction_df.empty:
        return pd.DataFrame(columns=["source", "target"])

    if "source" in interaction_df.columns and "target" in interaction_df.columns:
        edges = interaction_df[["source", "target"]].copy()
    else:
        if interaction_df.shape[1] < 2:
            return pd.DataFrame(columns=["source", "target"])
        edges = pd.DataFrame({
            "source": interaction_df.iloc[:, -2].astype(str),
            "target": interaction_df.iloc[:, -1].astype(str),
        })

    edges = edges.dropna()
    edges["source"] = edges["source"].astype(str).str.strip()
    edges["target"] = edges["target"].astype(str).str.strip()
    edges = edges[
        (edges["source"] != "")
        & (edges["target"] != "")
    ]
    return edges.reset_index(drop=True)

def create_and_analyze_network(
    edges: pd.DataFrame,
    net_name: str,
    directed: bool = False,
) -> int:
    """
    Create a Cytoscape network WITHOUT py4cytoscape's
    create_network_from_data_frames() helper.

    Reason:
    create_network_from_data_frames() automatically applies the default
    visual style (vizmap) and preferred layout. Pred-MutPDI only needs
    Analyzer topology tables, so those visualization steps are unnecessary
    and can trigger "_delay_until_stable(): Timeout trying to apply vizmap".

    Here we create the network directly from CytoscapeJS JSON via CyREST,
    wait briefly for the network model/table to become available, set it
    current, and run Analyzer. No style or layout is applied.
    """
    if edges is None or edges.empty:
        raise ValueError("Cannot analyze an empty edge table.")

    edge_df = edges[["source", "target"]].copy()
    edge_df["source"] = edge_df["source"].astype(str)
    edge_df["target"] = edge_df["target"].astype(str)

    node_ids = pd.unique(
        pd.concat(
            [edge_df["source"], edge_df["target"]],
            ignore_index=True,
        )
    ).tolist()

    json_nodes = [
        {
            "data": {
                "id": str(node_id),
                "name": str(node_id),
            }
        }
        for node_id in node_ids
    ]

    json_edges = []
    for i, row in edge_df.iterrows():
        source = str(row["source"])
        target = str(row["target"])
        interaction = "interacts with"
        json_edges.append(
            {
                "data": {
                    "id": f"e{i}",
                    "name": f"{source} ({interaction}) {target}",
                    "source": source,
                    "target": target,
                    "interaction": interaction,
                }
            }
        )

    cytoscapejs = {
        "data": {"name": net_name},
        "elements": {
            "nodes": json_nodes,
            "edges": json_edges,
        },
    }

    # This lower-level helper only POSTs the network JSON.
    # It does NOT run "vizmap apply" or a layout.
    net_suid = p4c.networks.create_network_from_cytoscapejs(
        cytoscapejs,
        title=net_name,
        collection="Pred-MutPDI_internal",
    )

    # Cytoscape updates its model asynchronously. Wait only for the network
    # object/table to become accessible; do not wait for any visual style.
    last_error = None
    for _ in range(50):
        try:
            p4c.networks.get_network_suid(net_suid)
            break
        except Exception as exc:
            last_error = exc
            time.sleep(0.1)
    else:
        raise RuntimeError(
            f"Cytoscape network model did not become ready: {last_error}"
        )

    try:
        p4c.networks.set_current_network(net_suid)
    except Exception:
        pass

    try:
        p4c.analyze_network(
            network=net_suid,
            directed=directed,
        )
    except TypeError:
        p4c.networks.set_current_network(net_suid)
        p4c.analyze_network()

    return net_suid

def extract_element(atom_name: str) -> str:
    atom_name = str(atom_name).strip()
    if "_" in atom_name:
        last = atom_name.split("_")[-1]
        if last:
            return last[0].upper()
    return "Unrecognized"

def classify_interaction(source: str, target: str) -> str:
    src = extract_element(source)
    tgt = extract_element(target)

    if "Unrecognized" in (src, tgt):
        return "Unrecognized"

    pair = {src, tgt}

    if pair in [{"H", "O"}, {"H", "N"}]:
        return "H–O/H–N"

    if pair in [{"N", "O"}, {"C", "O"}, {"C", "N"}]:
        return "N–O/C–O/C–N"

    return "Other"

def build_interaction_frames(wt_pdb: Path, mut_pdb: Path, chain: str, mut: str) -> dict:
    _, pos, _ = parse_mut(mut)
    return {
        'inner_w': extract_aa_aa_contacts(wt_pdb, chain, pos, cutoff=5.0),
        'inner_m': extract_aa_aa_contacts(mut_pdb, chain, pos, cutoff=5.0),
        'inter_w': extract_aa_na_contacts(wt_pdb, chain, pos, cutoff=10.0),
        'inter_m': extract_aa_na_contacts(mut_pdb, chain, pos, cutoff=10.0),
    }


def _resolve_edge_endpoints(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """CyREST may expose node SUIDs, omit endpoints, or return named endpoints."""
    edges = edges.copy()
    id_names = {}
    if 'SUID' in nodes and 'name' in nodes:
        id_names = {str(k): str(v) for k, v in zip(nodes['SUID'], nodes['name'])}
    if 'source' in edges and 'target' in edges:
        for column in ('source', 'target'):
            edges[column] = edges[column].map(lambda value: id_names.get(str(value), str(value)))
        if all('_' in value for col in ('source', 'target') for value in edges[col]):
            return edges
    name_column = next((c for c in ('name', 'shared name') if c in edges), None)
    if name_column:
        parsed = edges[name_column].astype(str).str.extract(r'^(.*?)\s+\([^)]*\)\s+(.*?)$')
        if parsed.notna().all().all():
            edges['source'], edges['target'] = parsed[0], parsed[1]
            return edges
    raise ValueError('Cannot map Cytoscape edge endpoints back to atom names; contact categories would be invalid.')


def analyze_interaction_frame(interaction_df: pd.DataFrame, net_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    edges = edge_frame_from_interactions(interaction_df)
    if edges.empty:
        return pd.DataFrame(), pd.DataFrame()
    api = _connect_cytoscape()
    net_suid = None
    try:
        net_suid = create_and_analyze_network(edges, net_name, directed=False)
        nodes = pd.DataFrame(api.tables.get_table_columns('node', network=net_suid))
        edge_table = pd.DataFrame(api.tables.get_table_columns('edge', network=net_suid))
        if nodes.empty or edge_table.empty:
            raise ValueError('Analyzer returned an empty table for a nonempty graph.')
        edge_table = _resolve_edge_endpoints(nodes, edge_table)
        if 'EdgeBetweenness' not in edge_table:
            raise ValueError('Analyzer output lacks EdgeBetweenness.')
        return nodes, edge_table
    except Exception as exc:
        raise RuntimeError(f'Cytoscape analysis failed for {net_name}; no zero-filled prediction was produced. '
                           f'Underlying error: {exc}') from exc
    finally:
        if net_suid is not None:
            try:
                api.networks.delete_network(net_suid)
            except Exception:
                pass


def classify_node_type(name: str) -> str:
    prefix = str(name).split('_')[0].strip()
    if prefix in AA3:
        return 'amino'
    if prefix in NT_NAMES:
        return 'nuc'
    return 'unknown'


def safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors='coerce').dropna()
    if np.isinf(values.to_numpy(dtype=float)).any():
        raise ValueError('Infinite topology statistic.')
    return float(values.mean()) if len(values) else 0.0


def summarize_nodes_df(df: pd.DataFrame, network: str, state: str) -> dict[str, float]:
    out = {}
    if network not in {'inner','inter'} or state not in {'w','m'}:
        raise ValueError('Invalid network or state label.')
    if df.empty:
        aa = na = pd.DataFrame()
    else:
        if 'name' not in df:
            raise ValueError('Cytoscape node table lacks atom names.')
        df = df.copy()
        df['__type__'] = df['name'].map(classify_node_type)
        if (df['__type__'] == 'unknown').any():
            raise ValueError('Unknown node types in DNA graph.')
        aa, na = df[df['__type__']=='amino'], df[df['__type__']=='nuc']
        required = ({'ClosenessCentrality','BetweennessCentrality',
                     'NumberOfUndirectedEdges','NeighborhoodConnectivity'} if network == 'inner'
                    else {'ClosenessCentrality','Stress','BetweennessCentrality',
                          'Radiality','TopologicalCoefficient'})
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f'Missing topology statistics: {sorted(missing)}')
    out[f'n_amino_{network}_{state}'] = float(len(aa))
    out[f'n_nuc_{network}_{state}'] = float(len(na))
    for feature in NODE_FEATURES:
        a = safe_mean(aa[feature]) if len(aa) and feature in aa else 0.0
        n = safe_mean(na[feature]) if len(na) and feature in na else 0.0
        if network == 'inner':
            out[f'{feature}_AAinner_{state}'] = a
        else:
            out[f'{feature}_AAinter_{state}'] = a
            out[f'{feature}_NAinter_{state}'] = n
            out[f'{feature}_Weightedinter_{state}'] = 0.6*a + 0.4*n
    return out


def summarize_edges_df(df: pd.DataFrame, suffix: str) -> dict[str, float]:
    types = ('H–O/H–N','N–O/C–O/C–N','Other')
    out = {f'Total_Interactions_{suffix}': float(len(df))}
    if not df.empty:
        required = {'source','target','EdgeBetweenness'}
        if not required.issubset(df.columns):
            raise ValueError('Incomplete Cytoscape edge table.')
        df = df.copy()
        df['__type__'] = [classify_interaction(s,t) for s,t in zip(df['source'],df['target'])]
        if (df['__type__']=='Unrecognized').any():
            raise ValueError('Unrecognized atom name while classifying edges.')
    means = []
    for group in types:
        selected = df[df['__type__']==group] if not df.empty else df
        count = len(selected)
        mean = safe_mean(selected['EdgeBetweenness']) if count else 0.0
        out[f'{group}_Count_{suffix}'] = float(count)
        out[f'{group}_Mean_{suffix}'] = mean
        out[f'{group}_Ratio_{suffix}'] = count/len(df) if len(df) else 0.0
        means.append(mean)
    out[f'Network_Evaluation_{suffix}'] = 0.6*means[0] + 0.3*means[1] + 0.1*means[2]
    return out


def combine_state_features(per_state: dict[str, float]) -> dict[str, float]:
    out = dict(per_state)
    for feature in NODE_FEATURES:
        for role, delta_role in [('AAinter','AA_inter'),('NAinter','NA_inter'),
                                 ('Weightedinter','Weighted_inter'),('AAinner','AA_inner')]:
            out[f'Δ_{feature}_{delta_role}'] = out[f'{feature}_{role}_m'] - out[f'{feature}_{role}_w']
    for network in ('inner','inter'):
        bases = ['Total_Interactions','Network_Evaluation']
        bases += [f'{kind}_{stat}' for kind in ('H–O/H–N','N–O/C–O/C–N','Other')
                  for stat in ('Count','Mean','Ratio')]
        for name in bases:
            out[f'Δ_{name}_{network}'] = out[f'{name}_{network}_m'] - out[f'{name}_{network}_w']
    missing = set(FEATURE_ORDER) - set(out)
    if missing:
        raise KeyError(f'Unimplemented DNA network features: {sorted(missing)}')
    result = {name: float(out[name]) for name in FEATURE_ORDER}
    if len(result) != 36 or not all(math.isfinite(v) for v in result.values()):
        raise ValueError('Invalid DNA network feature vector.')
    return result


def compute_network_features(pdb_id: str, chain: str, mut: str, *,
                             wt_pdb: Path, mut_pdb: Path) -> dict[str, float]:
    frames = build_interaction_frames(Path(wt_pdb), Path(mut_pdb), chain, mut)
    run_id = uuid.uuid4().hex[:10]
    features = {}
    for suffix, frame in frames.items():
        network, state = suffix.rsplit('_',1)
        nodes, edges = analyze_interaction_frame(frame, safe_filename(f'{pdb_id}_{chain}_{mut}_{suffix}_{run_id}'))
        features.update(summarize_nodes_df(nodes, network, state))
        features.update(summarize_edges_df(edges, suffix))
    return combine_state_features(features)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    parser.error('Internal component. Use predict_DNA.sh or validate_features_DNA.py.')

if __name__ == '__main__':
    main()

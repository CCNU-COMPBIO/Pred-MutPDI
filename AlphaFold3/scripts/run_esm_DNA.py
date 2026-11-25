#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import gc
import os
from pathlib import Path
import numpy as np
from common_DNA import AA20, inspect_complex


def entropy_from_selected_probabilities(probabilities) -> float:
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (20,) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('Expected 20 finite, nonnegative amino-acid probabilities.')
    return float(-np.sum(p*np.log2(p+1e-12)))


def compute_hi(model, alphabet, seq: str, idx0: int, device) -> float:
    import torch
    masked = seq[:idx0] + '<mask>' + seq[idx0+1:]
    _, _, tokens = alphabet.get_batch_converter()([('protein',masked)])
    tokens = tokens.to(device)
    with torch.no_grad():
        logits = model(tokens)['logits'][0,idx0+1]
        probs = torch.softmax(logits, dim=-1)
        p = probs[[alphabet.get_idx(a) for a in AA20]].cpu().numpy()
    # Preserve the original float32 arithmetic when probabilities are float32.
    return float(-np.sum(p*np.log2(p+1e-12)))


def compute_esm_feature(pdb_id: str, chain: str, mut: str, *, wt_pdb: Path) -> float:
    info = inspect_complex(wt_pdb, chain, mut)
    target = next(c for c in info['chains'] if c['original_chain']==chain)
    import esm
    import torch
    checkpoint = os.environ.get('PRED_MUTPDI_ESM_CHECKPOINT')
    if checkpoint:
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(checkpoint)
    else:
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    device = torch.device(os.environ.get('PRED_MUTPDI_ESM_DEVICE',
                                         'cuda' if torch.cuda.is_available() else 'cpu'))
    try:
        model = model.to(device).eval()
        return compute_hi(model, alphabet, target['sequence'], info['mutation_index0'], device)
    finally:
        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    parser.error('Internal component. Use predict_DNA.sh.')

if __name__ == '__main__':
    main()

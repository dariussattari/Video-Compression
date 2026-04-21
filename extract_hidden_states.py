"""
Extract hidden states from a trained genomic compressor.

Runs chr22 through the model and saves the final transformer layer's
hidden representations for downstream SAE interpretability analysis.
"""

import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from model import DecoderOnlyTransformer
from data import GenomeDataset

# ── Configuration ──
CHECKPOINT_PATH = 'checkpoints/best_model.pt'
CHR22_TOKEN_PATH = 'data/chr22.npy'
OUTPUT_PATH = 'chr22_hidden_states.pt'

EXTRACT_LAYER = 11
CHUNK_SIZE = 2048
BATCH_SIZE = 16

VOCAB_SIZE = 4096
EMBED_DIM = 512
NUM_HEADS = 8
NUM_LAYERS = 12

SEED = 42


# ── Setup ──
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'  GPU:    {torch.cuda.get_device_name(0)}')
        print(f'  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    return device


# ── Model ──
def build_model(device):
    model = DecoderOnlyTransformer(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        bias=True,
        tie_weights=True,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f'Model: {NUM_LAYERS}L / {EMBED_DIM}d / {NUM_HEADS}H ({num_params:,} params)')
    return model


def load_checkpoint(path, model, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    bpb = checkpoint.get('val_bpb', None)
    epoch = checkpoint.get('epoch', None)
    print(f'Loaded checkpoint: epoch {epoch}, val bpb {bpb:.4f}' if bpb else f'Loaded checkpoint.')


# ── Data ──
def build_loader(token_path):
    tokens = np.load(token_path)
    chr22_tokens = {'chr22': tokens}

    dataset = GenomeDataset(
        chr22_tokens,
        ['chr22'],
        seq_len=CHUNK_SIZE,
        stride=CHUNK_SIZE,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    print(f'Chr22: {len(tokens):,} tokens → {len(dataset):,} chunks → {len(loader):,} batches')
    return loader


# ── Extraction ──
def extract_hidden_states(model, loader, layer, device):
    model.eval()
    all_hidden_states = []

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(loader):
            x = x.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                output = model(x, return_hidden_layer=layer)

            all_hidden_states.append(output.hidden_state.cpu())

            if batch_idx % 50 == 0:
                print(f'  Batch {batch_idx:>4} / {len(loader)}')

    hidden_states = torch.cat(all_hidden_states, dim=0).view(-1, EMBED_DIM)
    print(f'  Extracted: {hidden_states.shape}')
    return hidden_states


# ── Main ──
if __name__ == '__main__':
    set_seed(SEED)
    device = get_device()

    model = build_model(device)
    load_checkpoint(CHECKPOINT_PATH, model, device)
    loader = build_loader(CHR22_TOKEN_PATH)

    print(f'\nExtracting layer {EXTRACT_LAYER} hidden states')
    hidden_states = extract_hidden_states(model, loader, EXTRACT_LAYER, device)

    torch.save(hidden_states, OUTPUT_PATH)
    print(f'\nSaved {hidden_states.shape} → {OUTPUT_PATH}')
    print(f'Size: {hidden_states.nbytes / 1e9:.2f} GB')
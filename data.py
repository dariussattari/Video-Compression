import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import encode_6mer


def load_seq(file):
    with open(file, "r") as f:
        sequence = f.read()
    return sequence


def tokenize_chromosomes(path):
    for filename in os.listdir(path):
        if not filename.endswith(".txt"):
            continue

        input_path = os.path.join(path, filename)
        sequence = load_seq(input_path)
        tokens = encode_6mer(sequence)

        output_name = filename.replace(".txt", ".npy")
        output_path = os.path.join(path, output_name)
        np.save(output_path, tokens)

def load_tokens(path):
    chromosomes_tokens = {}

    for filename in sorted(os.listdir(path)):
        if not filename.endswith(".npy"):
            continue

        input_path = os.path.join(path, filename)
        tokens = np.load(input_path)

        chr_name = filename.replace(".npy", "")
        chromosomes_tokens[chr_name] = tokens

    return chromosomes_tokens


class GenomeDataset(Dataset):
    def __init__(self, chromosomes_tokens, chromosome_names, seq_len=512, stride=512):
        self.chromosomes_tokens = chromosomes_tokens
        self.chromosome_names = chromosome_names
        self.seq_len = seq_len
        self.stride = stride
        self.index_map = []

        for chr_name in self.chromosome_names:
            tokens = self.chromosomes_tokens[chr_name]
            n_tokens = len(tokens)

            max_start = n_tokens - self.seq_len - 1
            if max_start < 0:
                continue

            for start in range(0, max_start + 1, self.stride):
                self.index_map.append((chr_name, start))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        chr_name, start = self.index_map[idx]
        tokens = self.chromosomes_tokens[chr_name]

        x = tokens[start : start + self.seq_len]
        y = tokens[start + 1 : start + self.seq_len + 1]

        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)

        return x, y

def create_dataloaders(path, seq_len=512, stride=512, batch_size=32):
    chromosomes_tokens = load_tokens(path)

    train_chromosomes = [c for c in chromosomes_tokens if c != "chr22"]
    val_chromosomes = ["chr22"]

    train_dataset = GenomeDataset(
        chromosomes_tokens,
        train_chromosomes,
        seq_len=seq_len,
        stride=stride
    )

    val_dataset = GenomeDataset(
        chromosomes_tokens,
        val_chromosomes,
        seq_len=seq_len,
        stride=stride
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader
import torch
import torch.nn.functional as F
import numpy as np
import constriction
import math
import time
from model import DecoderOnlyTransformer

def encode(model, tokens, device, seq_len=512):
    num_tokens = len(tokens)
    num_predictions = num_tokens - 1

    encoder = constriction.stream.queue.RangeEncoder()
    start_time = time.time()

    with torch.no_grad():
        for pos in range(num_predictions):
            chunk_start = (pos // seq_len) * seq_len
            chunk_end = pos + 1
            chunk_tokens = tokens[chunk_start:chunk_end]

            chunk_tensor = torch.tensor(
                chunk_tokens, dtype=torch.long
            ).unsqueeze(0).to(device)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                output = model(chunk_tensor)
                logits = output.logits

            probs = F.softmax(logits[:, -1, :].float(), dim=-1).squeeze(0)
            probs = probs.cpu().numpy().astype(np.float64)
            probs = probs / probs.sum()

            target = int(tokens[pos + 1])
            entropy_model = constriction.stream.model.Categorical(
                probs, perfect=False
            )
            encoder.encode(np.array([target], dtype=np.int32), entropy_model)

            if pos % 10000 == 0:
                elapsed = time.time() - start_time
                tokens_per_sec = (pos + 1) / elapsed if elapsed > 0 else 0
                print(
                    f"  Encoded {pos+1:>10,} / {num_predictions:,} "
                    f"({100*(pos+1)/num_predictions:.1f}%) | "
                    f"{tokens_per_sec:,.0f} tok/s"
                )

    compressed = encoder.get_compressed()
    num_bits = len(compressed) * 32
    num_bases = num_predictions * 6
    bpb = num_bits / num_bases

    elapsed = time.time() - start_time
    print(f"\nEncoding complete:")
    print(f"  Time:       {elapsed/60:.1f} min")
    print(f"  Compressed: {num_bits:,} bits")
    print(f"  Ratio:      {bpb:.4f} bits per base")

    return compressed, num_bits

#faster, compatible with decompressor
def encode_with_cache(model, tokens, device, seq_len=512):
    num_tokens = len(tokens)
    num_predictions = num_tokens - 1

    encoder = constriction.stream.queue.RangeEncoder()
    start_time = time.time()

    past_kvs = None

    with torch.no_grad():
        for pos in range(num_predictions):
            # Reset cache at chunk boundaries to limit memory growth
            if pos % seq_len == 0:
                past_kvs = None
                # Feed all tokens in this chunk up to current position
                chunk_start = pos
                chunk_end = pos + 1
                chunk_tokens = tokens[chunk_start:chunk_end]
            else:
                # Feed only the latest token, reuse cached keys/values
                chunk_tokens = tokens[pos:pos + 1]

            chunk_tensor = torch.tensor(
                chunk_tokens, dtype=torch.long
            ).unsqueeze(0).to(device)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                output = model(
                    chunk_tensor,
                    past_kvs=past_kvs,
                    use_cache=True,
                )
                logits = output.logits
                past_kvs = output.past_kvs

            probs = F.softmax(logits[:, -1, :].float(), dim=-1).squeeze(0)
            probs = probs.cpu().numpy().astype(np.float64)
            probs = probs / probs.sum()

            target = int(tokens[pos + 1])
            entropy_model = constriction.stream.model.Categorical(
                probs, perfect=False
            )
            encoder.encode(np.array([target], dtype=np.int32), entropy_model)

            if pos % 10000 == 0:
                elapsed = time.time() - start_time
                tokens_per_sec = (pos + 1) / elapsed if elapsed > 0 else 0
                print(
                    f"  Encoded {pos+1:>10,} / {num_predictions:,} "
                    f"({100*(pos+1)/num_predictions:.1f}%) | "
                    f"{tokens_per_sec:,.0f} tok/s"
                )

    compressed = encoder.get_compressed()
    num_bits = len(compressed) * 32
    num_bases = num_predictions * 6
    bpb = num_bits / num_bases

    elapsed = time.time() - start_time
    print(f"\nEncoding complete:")
    print(f"  Time:       {elapsed/60:.1f} min")
    print(f"  Compressed: {num_bits:,} bits")
    print(f"  Ratio:      {bpb:.4f} bits per base")

    return compressed, num_bits


#fastest. cannot be decompressed
def encode_fast(model, tokens, device, seq_len=512):
    num_tokens = len(tokens)
    num_predictions = num_tokens - 1

    encoder = constriction.stream.queue.RangeEncoder()
    start_time = time.time()
    encoded_count = 0

    with torch.no_grad():
        for chunk_start in range(0, num_tokens - 1, seq_len):
            chunk_end = min(chunk_start + seq_len, num_tokens)
            chunk = torch.tensor(
                tokens[chunk_start:chunk_end], dtype=torch.long
            ).unsqueeze(0).to(device)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                output = model(chunk)
                logits = output.logits.squeeze(0).float()

            # logits[i] predicts token at chunk_start+i+1
            # use all positions except the last (nothing to predict against)
            num_preds = min(chunk_end - chunk_start - 1, num_tokens - chunk_start - 1)
            probs = F.softmax(logits[:num_preds], dim=-1).cpu().numpy().astype(np.float64)
            targets = tokens[chunk_start + 1:chunk_start + 1 + num_preds].astype(np.int32)

            for i in range(len(targets)):
                p = probs[i]
                p = p / p.sum()
                entropy_model = constriction.stream.model.Categorical(
                    p, perfect=False
                )
                encoder.encode(np.array([targets[i]], dtype=np.int32), entropy_model)

            encoded_count += len(targets)
            elapsed = time.time() - start_time
            tokens_per_sec = encoded_count / elapsed if elapsed > 0 else 0
            print(
                f"  Encoded {encoded_count:>10,} / {num_predictions:,} "
                f"({100 * encoded_count / num_predictions:.1f}%) | "
                f"{tokens_per_sec:,.0f} tok/s"
            )

    compressed = encoder.get_compressed()
    num_bits = len(compressed) * 32
    num_bases = num_predictions * 6
    bpb = num_bits / num_bases

    elapsed = time.time() - start_time
    print(f"\nEncoding complete:")
    print(f"  Time:       {elapsed/60:.1f} min")
    print(f"  Compressed: {num_bits:,} bits ({num_bits / 8 / 1e6:.2f} MB)")
    print(f"  Ratio:      {bpb:.4f} bits per base")

    return compressed, num_bits
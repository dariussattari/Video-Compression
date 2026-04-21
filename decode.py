import torch
import torch.nn.functional as F
import numpy as np
import constriction
import math
import time
from model import DecoderOnlyTransformer


def decode(model, compressed, num_tokens, device, seq_len=512):
    """
    Decompress a bitstream back to tokens using the model's predictions.
    
    Args:
        model: same trained model used for encoding (on device, eval mode)
        compressed: numpy array of uint32 (the compressed bitstream)
        num_tokens: total number of tokens to recover (original sequence length)
        device: torch device
        seq_len: max context length before resetting (matches encoding chunks)
    
    Returns:
        decoded_tokens: numpy array of recovered token IDs
    """
    num_predictions = num_tokens - 1

    decoder = constriction.stream.queue.RangeDecoder(compressed)

    # We need the first token to start — it's not compressed,
    # the encoder and decoder must agree on it.
    # We'll handle this by passing it separately.
    # For now, assume we know tokens[0].
    decoded = []

    start_time = time.time()

    with torch.no_grad():
        for start in range(0, num_predictions, seq_len):
            end = min(start + seq_len, num_predictions)
            chunk_len = end - start

            # Build the chunk from already-decoded tokens
            if start == 0:
                # First chunk: we need tokens[0] to start
                # This must be stored/transmitted separately
                raise ValueError(
                    "First token must be provided — see decode_full()"
                )

            chunk_start = start
            chunk_tokens = decoded[chunk_start:chunk_start + chunk_len]
            chunk_tensor = torch.tensor(
                chunk_tokens, dtype=torch.long
            ).unsqueeze(0).to(device)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                output = model(chunk_tensor)
                logits = output.logits

            probs = F.softmax(logits.float(), dim=-1).squeeze(0)
            probs = probs.cpu().numpy().astype(np.float64)

            for i in range(chunk_len):
                prob_dist = probs[i]
                prob_dist = prob_dist / prob_dist.sum()

                entropy_model = constriction.stream.model.Categorical(
                    prob_dist, perfect=False
                )
                symbol = decoder.decode(entropy_model)
                decoded.append(int(symbol))

            # Progress
            elapsed = time.time() - start_time
            tokens_done = len(decoded)
            tokens_per_sec = tokens_done / elapsed if elapsed > 0 else 0
            if start % (seq_len * 100) == 0:
                print(
                    f"  Decoded {tokens_done:>10,} / {num_predictions:,} "
                    f"({100*tokens_done/num_predictions:.1f}%) | "
                    f"{tokens_per_sec:,.0f} tok/s"
                )

    elapsed = time.time() - start_time
    print(f"\nDecoding complete:")
    print(f"  Time:   {elapsed/60:.1f} min")
    print(f"  Tokens: {len(decoded):,}")

    return np.array(decoded, dtype=np.int64)


def decode_full(model, compressed, first_token, num_tokens, device, seq_len=512):
    num_predictions = num_tokens - 1
    decoder = constriction.stream.queue.RangeDecoder(compressed)
    decoded = [int(first_token)]

    start_time = time.time()

    with torch.no_grad():
        for pos in range(num_predictions):
            # Determine which chunk this position belongs to
            chunk_start = (pos // seq_len) * seq_len
            chunk_end = min(chunk_start + pos + 1, len(decoded))
            chunk_tokens = decoded[chunk_start:chunk_end]

            chunk_tensor = torch.tensor(
                chunk_tokens, dtype=torch.long
            ).unsqueeze(0).to(device)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                output = model(chunk_tensor)
                logits = output.logits

            # Take the LAST position's prediction only
            probs = F.softmax(logits[:, -1, :].float(), dim=-1).squeeze(0)
            probs = probs.cpu().numpy().astype(np.float64)
            probs = probs / probs.sum()

            entropy_model = constriction.stream.model.Categorical(
                probs, perfect=False
            )
            symbol = decoder.decode(entropy_model)
            decoded.append(int(symbol))

            if pos % 10000 == 0:
                elapsed = time.time() - start_time
                tokens_per_sec = (pos + 1) / elapsed if elapsed > 0 else 0
                print(
                    f"  Decoded {pos+1:>10,} / {num_predictions:,} "
                    f"({100*(pos+1)/num_predictions:.1f}%) | "
                    f"{tokens_per_sec:,.0f} tok/s"
                )

    elapsed = time.time() - start_time
    decoded = np.array(decoded, dtype=np.int64)
    print(f"\nDecoding complete: {elapsed/60:.1f} min")

    return decoded

def decode_with_cache(model, compressed, first_token, num_tokens, device, seq_len=512):
    num_predictions = num_tokens - 1

    decoder = constriction.stream.queue.RangeDecoder(compressed)
    decoded = [int(first_token)]

    past_kvs = None
    start_time = time.time()

    with torch.no_grad():
        for pos in range(num_predictions):
            # Reset cache at chunk boundaries — matches encoder
            if pos % seq_len == 0:
                past_kvs = None
                chunk_start = pos
                chunk_tokens = [decoded[chunk_start]]
            else:
                chunk_tokens = [decoded[-1]]

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

            entropy_model = constriction.stream.model.Categorical(
                probs, perfect=False
            )
            symbol = decoder.decode(entropy_model)
            decoded.append(int(symbol))

            if pos % 10000 == 0:
                elapsed = time.time() - start_time
                tokens_per_sec = (pos + 1) / elapsed if elapsed > 0 else 0
                print(
                    f"  Decoded {pos+1:>10,} / {num_predictions:,} "
                    f"({100*(pos+1)/num_predictions:.1f}%) | "
                    f"{tokens_per_sec:,.0f} tok/s"
                )

    elapsed = time.time() - start_time
    decoded = np.array(decoded, dtype=np.int64)
    print(f"\nDecoding complete: {elapsed/60:.1f} min")

    return decoded
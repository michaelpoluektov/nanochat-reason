from __future__ import annotations

"""
Engine for efficient inference of our models.

Everything works around token sequences:
- The user can send token sequences to the engine
- The engine returns the next token

Notes:
- The engine knows nothing about tokenization, it's purely token id sequences.

The whole thing is made as efficient as possible.
"""

from collections import deque
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
import signal
import warnings

import torch
import torch.nn.functional as F

from nanochat.common import compute_init
from nanochat.checkpoint_manager import load_model

# -----------------------------------------------------------------------------
# Calculator tool helpers
@contextmanager
def timeout(duration: int, formula: str) -> Iterator[None]:
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)

def eval_with_timeout(formula: str, max_time: int = 3) -> float | int | None:
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula)
    except Exception as e:
        signal.alarm(0)
        # print(f"Warning: Failed to eval {formula}, exception: {e}") # it's ok ignore wrong calculator usage
        return None

def use_calculator(expr: str) -> float | int | None:
    """Evaluate a math expression safely."""
    expr = expr.replace(",", "")
    if any([x not in "0123456789*+-/.() " for x in expr]): # for now disallow non-numeric chars
        return None
    if "**" in expr: # for now disallow power operator, could be very expensive
        return None
    return eval_with_timeout(expr)


def _normalize_samples(num_samples: int | Sequence[int], num_prompts: int) -> list[int]:
    if isinstance(num_samples, int):
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        return [num_samples] * num_prompts
    samples = list(num_samples)
    if len(samples) != num_prompts:
        raise ValueError("num_samples sequence must match number of prompts")
    if any(s <= 0 for s in samples):
        raise ValueError("num_samples values must be positive")
    return samples

# -----------------------------------------------------------------------------
class KVCache:
    """Works hand-in-hand with the GPT model and tracks per-row positions."""

    def __init__(self, batch_size: int, num_heads: int, seq_len: int, head_dim: int, num_layers: int):
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = seq_len
        self.kv_shape = (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
        self.kv_cache: torch.Tensor | None = None
        self.positions = torch.zeros(batch_size, dtype=torch.long)
        self.active_rows: tuple[int, ...] = tuple(range(batch_size))

    def reset(self) -> None:
        self.positions.zero_()

    def set_active_rows(self, rows: Sequence[int] | None) -> None:
        if rows is None:
            self.active_rows = tuple(range(self.batch_size))
            return
        rows = tuple(rows)
        if not rows:
            raise ValueError("Active rows cannot be empty")
        if any(r < 0 or r >= self.batch_size for r in rows):
            raise ValueError(f"Row index out of bounds in {rows}")
        self.active_rows = rows

    def get_active_positions(self, device: torch.device) -> torch.Tensor:
        if not self.active_rows:
            raise RuntimeError("Active rows are not set")
        idx = torch.tensor(self.active_rows, dtype=torch.long, device=device)
        return self.positions.to(device=device)[idx]

    def prefill(self, other: "KVCache", target_rows: Sequence[int] | None = None) -> None:
        assert other.kv_cache is not None, "Cannot prefill with an empty cache"
        for ix, (dim1, dim2) in enumerate(zip(self.kv_shape, other.kv_shape)):
            if ix in [0, 1, 3, 5]:
                assert dim1 == dim2, f"Dim {ix} mismatch: {dim1} != {dim2}"
            elif ix == 2:
                assert dim1 == dim2 or dim2 == 1, f"Batch dim mismatch: {dim1} != {dim2}"
            elif ix == 4:
                assert dim1 >= dim2, f"Seq len mismatch: {dim1} < {dim2}"
        dtype, device = other.kv_cache.dtype, other.kv_cache.device
        if self.kv_cache is None:
            self.kv_cache = torch.empty(self.kv_shape, dtype=dtype, device=device)
        elif self.kv_cache.dtype != dtype or self.kv_cache.device != device:
            raise ValueError("Prefill dtype/device mismatch")
        rows = tuple(range(self.batch_size)) if target_rows is None else tuple(target_rows)
        if not rows:
            raise ValueError("target_rows cannot be empty")
        if other.batch_size not in (1, len(rows)):
            raise ValueError("Source cache batch size must match target rows or be 1")
        for i, row in enumerate(rows):
            src_row = 0 if other.batch_size == 1 else i
            length = int(other.positions[src_row].item())
            self.positions[row] = length
            self.kv_cache[:, :, row, :, :length, :] = other.kv_cache[:, :, src_row, :, :length, :]

    def insert_kv(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        if self.kv_cache is None:
            self.kv_cache = torch.empty(self.kv_shape, dtype=k.dtype, device=k.device)
        rows = self.active_rows
        B, _, T_add, _ = k.size()
        if B != len(rows):
            raise ValueError(f"insert_kv expected {len(rows)} rows, got {B}")
        for batch_row, cache_row in enumerate(rows):
            t0 = int(self.positions[cache_row].item())
            t1 = t0 + T_add
            if t1 > self.kv_cache.size(4):
                t_needed = t1 + 1024
                t_needed = (t_needed + 1023) & ~1023
                additional_shape = list(self.kv_cache.shape)
                additional_shape[4] = t_needed - self.kv_cache.size(4)
                additional_cache = torch.empty(additional_shape, dtype=k.dtype, device=k.device)
                self.kv_cache = torch.cat([self.kv_cache, additional_cache], dim=4).contiguous()
                self.kv_shape = self.kv_cache.shape
            self.kv_cache[layer_idx, 0, cache_row, :, t0:t1, :] = k[batch_row]
            self.kv_cache[layer_idx, 1, cache_row, :, t0:t1, :] = v[batch_row]
        if layer_idx == self.num_layers - 1:
            for cache_row in rows:
                self.positions[cache_row] += T_add
        row_tensor = torch.tensor(rows, dtype=torch.long)
        lengths = self.positions[row_tensor]
        max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
        rows_list = list(rows)
        key_view = self.kv_cache[layer_idx, 0, rows_list, :, :max_len, :]
        value_view = self.kv_cache[layer_idx, 1, rows_list, :, :max_len, :]
        return key_view, value_view, lengths


# -----------------------------------------------------------------------------
@torch.inference_mode()
def sample_next_token(
    logits: torch.Tensor,
    rng: torch.Generator,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """Sample a single next token from given logits of shape (B, vocab_size). Returns (B, 1)."""
    assert temperature >= 0.0, "temperature must be non-negative"
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)

# -----------------------------------------------------------------------------

class RowState:
    # Per-row state tracking during generation
    def __init__(self, current_tokens: list[int] | None = None):
        self.current_tokens: list[int] = current_tokens or [] # Current token sequence for this row
        self.forced_tokens: deque[int] = deque() # Queue of tokens to force inject
        self.in_python_block: bool = False # Whether we are inside a python block
        self.python_expr_tokens: list[int] = [] # Tokens of the current python expression
        self.completed: bool = False # Whether this row has completed generation

class Engine:

    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer # needed for tool use

    @torch.inference_mode()
    def generate(
        self,
        tokens: list[int],
        num_samples: int = 1,
        max_tokens: int | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        seed: int = 42,
    ) -> Iterator[tuple[list[int], list[int]]]:
        """Same as generate, but does single prefill and then clones the KV cache."""
        assert isinstance(tokens, list) and isinstance(tokens[0], int), "expecting list of ints"
        device = self.model.get_device()
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        # Get the special tokens we need to coordinate the tool use state machine
        get_special = lambda s: self.tokenizer.encode_special(s)
        python_start = get_special("<|python_start|>")
        python_end = get_special("<|python_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>") # if sampled, ends row
        bos = self.tokenizer.get_bos_token_id() # if sampled, ends row

        # 1) Run a batch 1 prefill of the prompt tokens
        m = self.model.config
        kv_model_kwargs = {"num_heads": m.n_kv_head, "head_dim": m.n_embd // m.n_head, "num_layers": m.n_layer}
        kv_cache_prefill = KVCache(
            batch_size=1,
            seq_len=len(tokens),
            **kv_model_kwargs,
        )
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
        logits = logits[:, -1, :]
        next_ids = sample_next_token(logits, rng, temperature, top_k)  # (B, 1)
        sampled_tokens = next_ids[:, 0].tolist()

        # 2) Replicate the KV cache for each sample/row
        kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else self.model.config.sequence_len
        kv_cache_decode = KVCache(
            batch_size=num_samples,
            seq_len=kv_length_hint,
            **kv_model_kwargs,
        )
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill # no need to keep this memory around

        # 3) Initialize states for each sample
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

        # 4) Main generation loop
        num_generated = 0
        first_iteration = True
        while True:
            # Stop condition: we've reached max tokens
            if max_tokens is not None and num_generated >= max_tokens:
                break
            # Stop condition: all rows are completed
            if all(state.completed for state in row_states):
                break

            # Get sampled tokens - either from prefill or from forward pass
            if first_iteration:
                # Use the tokens we already sampled from prefill
                sampled_tokens = [sampled_tokens[0]] * num_samples  # Broadcast first token to all rows
                # TODO: we should sample a token for each row instead of broadcasting
                first_iteration = False
            else:
                # Forward the model and get the next token for each row
                logits = self.model.forward(ids, kv_cache=kv_cache_decode)  # (B, T, vocab_size)
                logits = logits[:, -1, :]  # (B, vocab_size) at last time step
                next_ids = sample_next_token(logits, rng, temperature, top_k)  # (B, 1)
                sampled_tokens = next_ids[:, 0].tolist()

            # Process each row: choose the next token, update state, optional tool use
            token_column = [] # contains the next token id along each row
            token_masks = [] # contains the mask (was it sampled (1) or forced (0)?) along each row
            for i, state in enumerate(row_states):
                # Select the next token in this row
                is_forced = len(state.forced_tokens) > 0 # are there tokens waiting to be forced in deque?
                token_masks.append(0 if is_forced else 1) # mask is 0 if forced, 1 if sampled
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                token_column.append(next_token)
                # Update the state of this row to include the next token
                state.current_tokens.append(next_token)
                # On <|assistant_end|> or <|bos|>, mark the row as completed
                if next_token == assistant_end or next_token == bos:
                    state.completed = True
                # Handle tool logic
                if next_token == python_start:
                    state.in_python_block = True
                    state.python_expr_tokens = []
                elif next_token == python_end and state.in_python_block:
                    state.in_python_block = False
                    if state.python_expr_tokens:
                        expr = self.tokenizer.decode(state.python_expr_tokens)
                        result = use_calculator(expr)
                        if result is not None:
                            result_tokens = self.tokenizer.encode(str(result))
                            state.forced_tokens.append(output_start)
                            state.forced_tokens.extend(result_tokens)
                            state.forced_tokens.append(output_end)
                    state.python_expr_tokens = []
                elif state.in_python_block:
                    state.python_expr_tokens.append(next_token)

            # Yield the token column
            yield token_column, token_masks
            num_generated += 1
            # Prepare ids for next iteration
            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)

    def generate_batch(self, tokens: list[int], num_samples: int | Sequence[int] = 1, **kwargs) -> tuple[list[list[int]], list[list[int]]]:
        sequences, masks = self.generate_multi_batch([tokens], num_samples, **kwargs)
        return sequences[0], masks[0]

    def generate_multi_batch(
        self,
        prompts: Sequence[list[int]],
        num_samples: int | Sequence[int] = 1,
        max_tokens: int | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        seed: int = 42,
    ) -> tuple[list[list[list[int]]], list[list[list[int]]]]:
        if not prompts:
            raise ValueError("prompts must be a non-empty sequence")

        prompts = [list(p) for p in prompts]
        samples_per_prompt = _normalize_samples(num_samples, len(prompts))
        total_rows = sum(samples_per_prompt)
        if total_rows == 0:
            raise ValueError("num_samples must allocate at least one row")

        device = self.model.get_device()
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        # Special tokens for tool handling
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        python_start = self.tokenizer.encode_special("<|python_start|>")
        python_end = self.tokenizer.encode_special("<|python_end|>")
        output_start = self.tokenizer.encode_special("<|output_start|>")
        output_end = self.tokenizer.encode_special("<|output_end|>")
        bos = self.tokenizer.get_bos_token_id()

        # Map each row to (prompt_idx, sample_idx)
        row_to_prompt: list[tuple[int, int]] = []
        prompt_row_ids: list[list[int]] = [[] for _ in prompts]
        row_idx = 0
        for prompt_idx, count in enumerate(samples_per_prompt):
            for sample_idx in range(count):
                row_to_prompt.append((prompt_idx, sample_idx))
                prompt_row_ids[prompt_idx].append(row_idx)
                row_idx += 1

        # Prepare KV cache for decoding
        m = self.model.config
        kv_kwargs = {"num_heads": m.n_kv_head, "head_dim": m.n_embd // m.n_head, "num_layers": m.n_layer}
        max_prompt_len = max(len(p) for p in prompts)
        if max_prompt_len >= m.sequence_len:
            raise ValueError("Prompt length exceeds model sequence length")
        max_decode_tokens = max_tokens if max_tokens is not None else m.sequence_len - max_prompt_len
        max_decode_tokens = max(1, min(max_decode_tokens, m.sequence_len - max_prompt_len))
        kv_length_hint = min(m.sequence_len, max_prompt_len + max_decode_tokens)
        decode_cache = KVCache(batch_size=total_rows, seq_len=kv_length_hint, **kv_kwargs)

        vocab_size = self.model.config.vocab_size
        initial_logits = torch.empty((total_rows, vocab_size), dtype=torch.float32, device=device)

        # Prefill each prompt individually and copy into shared cache
        for prompt_idx, tokens in enumerate(prompts):
            row_ids = prompt_row_ids[prompt_idx]
            if not row_ids:
                continue
            prompt_cache = KVCache(batch_size=1, seq_len=len(tokens), **kv_kwargs)
            prompt_cache.set_active_rows((0,))
            ids = torch.tensor([tokens], dtype=torch.long, device=device)
            logits = self.model.forward(ids, kv_cache=prompt_cache)[:, -1, :]
            decode_cache.prefill(prompt_cache, target_rows=row_ids)
            initial_logits[row_ids] = logits.expand(len(row_ids), -1)

        # Initialize per-row state
        row_states = [RowState(prompts[p_idx].copy()) for p_idx, _ in row_to_prompt]
        row_masks = [[0] * len(prompts[p_idx]) for p_idx, _ in row_to_prompt]
        completion_counts = [0] * total_rows
        pending_tokens: list[int | None] = [None] * total_rows

        def apply_token(row: int, token: int, sampled_mask: int) -> bool:
            state = row_states[row]
            appended = False
            if token not in (assistant_end, bos):
                state.current_tokens.append(token)
                row_masks[row].append(sampled_mask)
                appended = True
            if token == assistant_end or token == bos:
                state.completed = True
            if token == python_start:
                state.in_python_block = True
                state.python_expr_tokens = []
            elif token == python_end and state.in_python_block:
                state.in_python_block = False
                if state.python_expr_tokens:
                    expr = self.tokenizer.decode(state.python_expr_tokens)
                    result = use_calculator(expr)
                    if result is not None:
                        injected = self.tokenizer.encode(str(result))
                        state.forced_tokens.append(output_start)
                        state.forced_tokens.extend(injected)
                        state.forced_tokens.append(output_end)
                state.python_expr_tokens = []
            elif state.in_python_block:
                state.python_expr_tokens.append(token)
            return appended

        def emit_tokens(row_indices: Sequence[int], raw_tokens: Sequence[int]) -> None:
            for idx, row in enumerate(row_indices):
                state = row_states[row]
                is_forced = len(state.forced_tokens) > 0
                next_token = state.forced_tokens.popleft() if is_forced else raw_tokens[idx]
                appended = apply_token(row, next_token, 0 if is_forced else 1)
                if appended:
                    completion_counts[row] += 1
                if max_tokens is not None and completion_counts[row] >= max_tokens:
                    state.completed = True
                pending_tokens[row] = None if state.completed else next_token

        first_tokens = sample_next_token(initial_logits, rng, temperature, top_k)[:, 0].tolist()
        emit_tokens(range(total_rows), first_tokens)

        while True:
            active_rows = [i for i, token in enumerate(pending_tokens) if token is not None]
            if not active_rows:
                break
            ids = torch.tensor([pending_tokens[i] for i in active_rows], dtype=torch.long, device=device).unsqueeze(1)
            for row in active_rows:
                pending_tokens[row] = None
            decode_cache.set_active_rows(active_rows)
            logits = self.model.forward(ids, kv_cache=decode_cache)[:, -1, :]
            raw_tokens = sample_next_token(logits, rng, temperature, top_k)[:, 0].tolist()
            emit_tokens(active_rows, raw_tokens)

        # Assemble outputs per prompt/sample
        prompt_sequences: list[list[list[int]]] = [
            [
                list(row_states[row_idx].current_tokens)
                for row_idx in prompt_row_ids[prompt_idx]
            ]
            for prompt_idx in range(len(prompts))
        ]
        prompt_masks: list[list[list[int]]] = [
            [
                list(row_masks[row_idx])
                for row_idx in prompt_row_ids[prompt_idx]
            ]
            for prompt_idx in range(len(prompts))
        ]
        return prompt_sequences, prompt_masks


if __name__ == "__main__":
    """
    Quick inline test to make sure that the naive/slow model.generate function
    is equivalent to the faster Engine.generate function here.
    """
    import time
    # init compute
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    # load the model and tokenizer
    model, tokenizer, meta = load_model("base", device, phase="eval")
    bos_token_id = tokenizer.get_bos_token_id()
    # common hyperparameters
    kwargs = dict(max_tokens=64, temperature=0.0)
    # set the starting prompt
    prompt_tokens = tokenizer.encode("The chemical formula of water is", prepend=bos_token_id)
    # generate the reference sequence using the model.generate() function
    generated_tokens = []
    is_cuda = device.type == "cuda"
    if is_cuda:
        torch.cuda.synchronize()
    t0 = time.time()
    stream = model.generate(prompt_tokens, **kwargs)
    for token in stream:
        generated_tokens.append(token)
        chunk = tokenizer.decode([token])
        print(chunk, end="", flush=True)
    print()
    if is_cuda:
        torch.cuda.synchronize()
    t1 = time.time()
    print(f"Reference time: {t1 - t0:.2f}s")
    reference_ids = generated_tokens
    # generate tokens with Engine
    generated_tokens = []
    engine = Engine(model, tokenizer)
    stream = engine.generate(prompt_tokens, num_samples=1, **kwargs) # note: runs in fp32
    if is_cuda:
        torch.cuda.synchronize()
    t0 = time.time()
    for token_column, token_masks in stream:
        token = token_column[0] # only print out the first row
        generated_tokens.append(token)
        chunk = tokenizer.decode([token])
        print(chunk, end="", flush=True)
    print()
    if is_cuda:
        torch.cuda.synchronize()
    t1 = time.time()
    print(f"Engine time: {t1 - t0:.2f}s")
    # compare the two sequences
    for i in range(len(reference_ids)):
        if reference_ids[i] != generated_tokens[i]:
            print(f"Mismatch at {i}: {reference_ids[i]} != {generated_tokens[i]}")
            break
    print(f"Match: {reference_ids == generated_tokens}")

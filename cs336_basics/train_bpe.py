from __future__ import annotations

import heapq
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import BinaryIO
import regex as re

# GPT-2 regex pattern specified in Section 2.4 / 2.5
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    Boundaries are aligned with split_special_token.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if file_size == 0:
        return [0, 0]

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def _process_chunk(
    file_path: str | os.PathLike,
    start: int,
    end: int,
    special_tokens: list[str],
) -> Counter[tuple[bytes, ...]]:
    """
    Constant-memory streaming worker function:
    Streams the byte range [start, end) in 16MB buffers.
    Never loads large gigabyte files into RAM at once.
    """
    str_counts: Counter[str] = Counter()
    pat_compiled = re.compile(GPT2_SPLIT_PATTERN)

    primary_token_bytes = special_tokens[0].encode("utf-8") if special_tokens else None
    buffer_size = 16 * 1024 * 1024  # 16 MB streaming window

    with open(file_path, "rb") as f:
        f.seek(start)
        current_pos = start
        buffer = b""

        while current_pos < end:
            to_read = min(buffer_size, end - current_pos)
            chunk = f.read(to_read)
            if not chunk:
                break
            current_pos += len(chunk)
            buffer += chunk

            if primary_token_bytes:
                while True:
                    idx = buffer.find(primary_token_bytes)
                    if idx == -1:
                        break
                    doc_bytes = buffer[:idx]
                    buffer = buffer[idx + len(primary_token_bytes):]
                    if doc_bytes:
                        doc_text = doc_bytes.decode("utf-8", errors="ignore")
                        for match in pat_compiled.finditer(doc_text):
                            str_counts[match.group()] += 1
            else:
                last_newline = buffer.rfind(b"\n")
                if last_newline != -1:
                    line_block = buffer[:last_newline]
                    buffer = buffer[last_newline + 1:]
                    if line_block:
                        block_text = line_block.decode("utf-8", errors="ignore")
                        for match in pat_compiled.finditer(block_text):
                            str_counts[match.group()] += 1

        # Process any remaining bytes at chunk end
        if buffer:
            tail_text = buffer.decode("utf-8", errors="ignore")
            for match in pat_compiled.finditer(tail_text):
                str_counts[match.group()] += 1

    # Convert ONLY unique words to byte tuples (once at the very end)
    byte_counts: Counter[tuple[bytes, ...]] = Counter()
    for word_str, cnt in str_counts.items():
        word_bytes = word_str.encode("utf-8")
        if word_bytes:
            word_tuple = tuple(bytes([b]) for b in word_bytes)
            byte_counts[word_tuple] = cnt

    return byte_counts


class _DoublyLinkedListNode:
    __slots__ = ("token", "prev", "next")

    def __init__(self, token: bytes):
        self.token = token
        self.prev: _DoublyLinkedListNode | None = None
        self.next: _DoublyLinkedListNode | None = None


class _MaxPair:
    """Wrapper to break ties by choosing the lexicographically largest pair in a min-heap."""
    __slots__ = ("pair",)

    def __init__(self, pair: tuple[bytes, bytes]):
        self.pair = pair

    def __lt__(self, other: _MaxPair) -> bool:
        return self.pair > other.pair


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int | None = None,
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Given the path to an input corpus, train a byte-level BPE tokenizer and
    output its vocabulary and merges.
    """
    # 1. Initialize Base Vocabulary (256 bytes) + Special Tokens
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    existing_vocab_values = set(vocab.values())

    for special_tok in special_tokens:
        tok_bytes = special_tok.encode("utf-8")
        if tok_bytes not in existing_vocab_values:
            vocab[len(vocab)] = tok_bytes
            existing_vocab_values.add(tok_bytes)

    merges: list[tuple[bytes, bytes]] = []
    num_merges_target = vocab_size - len(vocab)
    if num_merges_target <= 0:
        return vocab, merges

    # 2. Pre-tokenization & Frequency Counting
    file_size = os.path.getsize(input_path)
    if num_processes is None:
        num_processes = min(4, os.cpu_count() or 1)

    word_counts: Counter[tuple[bytes, ...]] = Counter()

    if file_size > 1_000_000 and num_processes > 1 and special_tokens:
        print(f"Pre-tokenizing corpus using {num_processes} worker processes (streaming window)...")
        primary_split_token = special_tokens[0].encode("utf-8")
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_processes, primary_split_token)

        chunk_args = [
            (input_path, start, end, special_tokens)
            for start, end in zip(boundaries[:-1], boundaries[1:])
            if end > start
        ]

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [executor.submit(_process_chunk, *args) for args in chunk_args]
            for future in futures:
                word_counts.update(future.result())
    else:
        word_counts = _process_chunk(input_path, 0, file_size, special_tokens)

    print(f"Found {len(word_counts):,} unique pre-tokens. Indexing pairs for merging...")

    # 3. Build Indexed Data Structures
    words: list[list[_DoublyLinkedListNode]] = []
    word_freqs: list[int] = []
    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_to_words: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)

    for word_tuple, freq in word_counts.items():
        if len(word_tuple) < 2:
            continue
        word_idx = len(words)
        nodes = [_DoublyLinkedListNode(b) for b in word_tuple]
        for i in range(len(nodes) - 1):
            nodes[i].next = nodes[i + 1]
            nodes[i + 1].prev = nodes[i]
            pair = (nodes[i].token, nodes[i + 1].token)
            pair_counts[pair] += freq
            pair_to_words[pair].add(word_idx)

        words.append(nodes)
        word_freqs.append(freq)

    # 4. Initialize Priority Queue (O(N) heapify)
    pair_heap: list[tuple[int, _MaxPair]] = [
        (-cnt, _MaxPair(pair)) for pair, cnt in pair_counts.items() if cnt > 0
    ]
    heapq.heapify(pair_heap)

    print(f"Executing {num_merges_target:,} BPE merge steps...")

    # 5. Fast O(log N) BPE Merging Loop
    for merge_idx in range(num_merges_target):
        # Pop from heap with lazy validation (skip stale counts)
        best_pair: tuple[bytes, bytes] | None = None
        best_count = 0

        while pair_heap:
            neg_count, max_pair_obj = heapq.heappop(pair_heap)
            p = max_pair_obj.pair
            current_count = pair_counts.get(p, 0)
            if current_count > 0 and -neg_count == current_count:
                best_pair = p
                best_count = current_count
                break

        if best_pair is None or best_count <= 0:
            break

        del pair_counts[best_pair]
        first_tok, second_tok = best_pair
        merged_tok = first_tok + second_tok

        merges.append(best_pair)
        vocab[len(vocab)] = merged_tok

        if (merge_idx + 1) % 2000 == 0 or (merge_idx + 1) == num_merges_target:
            print(f"  Merge {merge_idx + 1:,} / {num_merges_target:,} (Vocab size: {len(vocab):,})")

        # Update words containing the best_pair
        affected_words = pair_to_words.pop(best_pair, set())

        for word_idx in affected_words:
            freq = word_freqs[word_idx]
            nodes = words[word_idx]

            curr: _DoublyLinkedListNode | None = None
            for node in nodes:
                if node.prev is None:
                    curr = node
                    break

            while curr is not None and curr.next is not None:
                if curr.token == first_tok and curr.next.token == second_tok:
                    next_node = curr.next
                    prev_node = curr.prev
                    after_node = next_node.next

                    # 1. Decrement old left pair & push updated count
                    if prev_node is not None:
                        old_left = (prev_node.token, curr.token)
                        pair_counts[old_left] -= freq
                        if pair_counts[old_left] <= 0:
                            del pair_counts[old_left]
                        else:
                            heapq.heappush(pair_heap, (-pair_counts[old_left], _MaxPair(old_left)))

                    # 2. Decrement old right pair & push updated count
                    if after_node is not None:
                        old_right = (next_node.token, after_node.token)
                        pair_counts[old_right] -= freq
                        if pair_counts[old_right] <= 0:
                            del pair_counts[old_right]
                        else:
                            heapq.heappush(pair_heap, (-pair_counts[old_right], _MaxPair(old_right)))

                    # 3. Merge nodes
                    curr.token = merged_tok
                    curr.next = after_node
                    if after_node is not None:
                        after_node.prev = curr

                    # 4. Increment new left pair & push new count
                    if prev_node is not None:
                        new_left = (prev_node.token, merged_tok)
                        pair_counts[new_left] += freq
                        pair_to_words[new_left].add(word_idx)
                        heapq.heappush(pair_heap, (-pair_counts[new_left], _MaxPair(new_left)))

                    # 5. Increment new right pair & push new count
                    if after_node is not None:
                        new_right = (merged_tok, after_node.token)
                        pair_counts[new_right] += freq
                        pair_to_words[new_right].add(word_idx)
                        heapq.heappush(pair_heap, (-pair_counts[new_right], _MaxPair(new_right)))

                else:
                    curr = curr.next

    return vocab, merges
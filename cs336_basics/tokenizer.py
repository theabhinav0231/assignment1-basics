from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
import regex as re

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(
            self,
            vocab: dict[int, bytes],
            merges: list[tuple[bytes, bytes]],
            special_tokens: list[str] | None = None,
    ):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.
        """
        self.vocab: dict[int, bytes] = dict(vocab)
        self.token_to_id: dict[bytes, int] = {b: i for i, b in self.vocab.items()}

        # Register special tokens if provided
        self.special_tokens = list(special_tokens) if special_tokens else []
        self.special_tokens_set = set(self.special_tokens)

        for special_tok in self.special_tokens:
            tok_bytes = special_tok.encode("utf-8")
            if tok_bytes not in self.token_to_id:
                new_id = len(self.vocab)
                self.vocab[new_id] = tok_bytes
                self.token_to_id[tok_bytes] = new_id

        # Compile regex to split on special tokens while preserving them as single tokens
        # Sort descending by length so longer special tokens match before substrings
        if self.special_tokens:
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            self.special_pattern = re.compile("(" + "|".join(re.escape(s) for s in sorted_specials) + ")")
        else:
            self.special_pattern = None

        # Pre-compiled GPT-2 regex pattern for pre-tokenization
        self.pat = re.compile(GPT2_SPLIT_PATTERN)

        # Index merges with priority rank for fast BPE lookup
        self.merges = list(merges)
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {
            pair: rank for rank, pair in enumerate(self.merges)
        }


    def _encode_word(self, word_bytes: bytes) -> list[int]:
        """
        Encode a single pre-token's UTF-8 bytes into token IDs using greedy BPE merge ranks.
        """
        if not word_bytes:
            return []

        # Start with individual byte tokens
        tokens: list[bytes] = [bytes([b]) for b in word_bytes]

        # Greedily apply lowest-rank merge until no more merges apply
        while len(tokens) >= 2:
            min_rank = float("inf")
            best_pair: tuple[bytes, bytes] | None = None

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is not None and rank < min_rank:
                    min_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            # Replace all non-overlapping occurrences of best_pair in the token list
            first_tok, second_tok = best_pair
            merged_token = first_tok + second_tok
            new_tokens: list[bytes] = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == first_tok and tokens[i + 1] == second_tok:
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return [self.token_to_id[t] for t in tokens]


    def encode(self, text: str) -> list[int]:
        """
        Encode an input string into a list of integer token IDs.
        """
        if not text:
            return []

        # If special tokens are registered, split while keeping special tokens intact
        if self.special_pattern is not None:
            segments = self.special_pattern.split(text)
        else:
            segments = [text]

        token_ids: list[int] = []
        for segment in segments:
            if not segment:
                continue

            if segment in self.special_tokens_set:
                token_ids.append(self.token_to_id[segment.encode("utf-8")])
            else:
                for match in self.pat.finditer(segment):
                    word_bytes = match.group().encode("utf-8")
                    token_ids.extend(self._encode_word(word_bytes))

        return token_ids


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g. lines from a large file),
        lazily yield integer token IDs without loading the entire content into memory.
        """
        for chunk in iterable:
            yield from self.encode(chunk)


    def decode(self, ids: list[int]) -> str:
        """
        Decode a list of integer token IDs into a Unicode string.
        Replaces malformed byte sequences with U+FFFD (the Unicode replacement character).
        """
        if not ids:
            return ""
        byte_sequence = b"".join(self.vocab[token_id] for token_id in ids)
        return byte_sequence.decode("utf-8", errors="replace")

    
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        """
        Construct a Tokenizer from serialized vocabulary and merges files.
        """
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)
        vocab = {
            int(k): v.encode("latin-1") if isinstance(v, str) else bytes(v)
            for k, v in raw_vocab.items()
        }
        vocab_set = set(vocab.values())
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\r\n")
                if not line:
                    continue
                found = False
                for i in range(len(line)):
                    if line[i] == " ":
                        p1 = line[:i].encode("latin-1")
                        p2 = line[i + 1 :].encode("latin-1")
                        if (p1 in vocab_set or not p1) and (p2 in vocab_set or not p2) and (p1 + p2 in vocab_set):
                            merges.append((p1, p2))
                            found = True
                            break
                if not found:
                    parts = line.split(" ")
                    if len(parts) == 2:
                        merges.append((parts[0].encode("latin-1"), parts[1].encode("latin-1")))
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
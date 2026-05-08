import regex as re
import json
from pathlib import Path
from tqdm import tqdm
from collections.abc import Iterable, Iterator
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" # GPT-2 regex 

class Tokenizer():
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab  # maps from int -> bytes
        self.merges = merges
        self.special_tokens = special_tokens

        self.inverse_vocab = {} # now maps from bytes -> int
        for key in self.vocab:
            value = self.vocab[key]
            self.inverse_vocab[value] = key # now maps from bytes -> int

        self.merge_ranks = {
            pair: rank
            for rank, pair in enumerate(self.merges)
        }

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath:str, special_tokens: list[str] | None = None):
        # path = Path("../tests/fixtures/gpt2_vocab.json")
        vocab = {}
        with open(vocab_filepath, 'r') as file:
            inverse_vocab = json.load(file)     # maps from bytes to int

        for byte_key in inverse_vocab:
            int_id = inverse_vocab[byte_key]
            byte_value = byte_key.encode('latin-1')
            vocab[int_id] = byte_value

        # merges_path = Path("../tests/fixtures/gpt2_merges.txt")
        merges = []
        with open(merges_filepath, 'r') as file:
            for line in file:
                # pair = re.split(pattern=' ', string= line.strip())
                pair = line.rstrip('\n').rsplit(" ", 1)
                if len(pair) == 2:
                    merges.append(tuple((pair[0].encode('latin-1'), pair[1].encode('latin-1'))))
        return cls(vocab, merges, special_tokens)
        

    # OLD - VERY SLOW IMPLEMENTATION    
    # def tokenize_pretoken(self, pretoken: list[bytes]) -> list[int]:
    #     old_pretoken = pretoken
    #     for merge in self.merges:
    #         new_pretoken = []
    #         i = 0 
    #         while i < len(old_pretoken) - 1:
    #             if old_pretoken[i] == merge[0] and old_pretoken[i + 1] == merge[1]: # We found the correct merge 
    #                 new_pretoken.append(old_pretoken[i] + old_pretoken[i + 1])
    #                 i += 2
    #             else:
    #                 new_pretoken.append(old_pretoken[i])
    #                 i += 1
    #         if i == len(old_pretoken) - 1:
    #             new_pretoken.append(old_pretoken[i]) 
    #         old_pretoken = new_pretoken

    #     output = []
    #     for byte in old_pretoken:
    #         id = self.inverse_vocab[byte]
    #         output.append(id)
    #     return output


    # NEW IMPLEMENTATION WITH A HELP OF CLAUDE
    def tokenize_pretoken(self, pretoken: list[bytes]) -> list[int]:
        while True:

            # Find all adjacent pairs
            pairs = [
                (pretoken[i], pretoken[i + 1])
                for i in range(len(pretoken) - 1)
            ]

            # Find mergeable pairs
            candidate_pairs = [
                (self.merge_ranks[pair], i, pair)
                for i, pair in enumerate(pairs)
                if pair in self.merge_ranks
            ]

            # No merges left
            if not candidate_pairs:
                break

            # Lowest rank = highest priority
            _, i, pair = min(candidate_pairs)

            # Perform merge
            pretoken = (
                pretoken[:i]
                + [pair[0] + pair[1]]
                + pretoken[i + 2:]
            )

        return [self.inverse_vocab[token] for token in pretoken]
    
    def encode(self, text:str) -> list[int]:
        if self.special_tokens is None:
            stripped_text = [text]
        else:
            pattern = "|".join(re.escape(t) for t in sorted(self.special_tokens, reverse=True))
            stripped_text = re.split(f'({pattern})', text) # list[str]


        BYTE_TABLE = [bytes([i]) for i in range(256)]
        output = []
        for segment in tqdm(stripped_text):
            if self.special_tokens is not None and segment in self.special_tokens:
                    token_id = self.inverse_vocab[segment.encode('utf-8')]
                    output.append(token_id)
            else:
                pretokens = re.findall(PAT, segment)
                for pretoken in pretokens:
                    # pretoken = list(bytes([b]) for b in pretoken.encode('utf-8'))
                    pretoken = [BYTE_TABLE[b] for b in pretoken.encode("utf-8")]
                    token_ids = self.tokenize_pretoken(pretoken=pretoken)
                    output.extend(token_ids)
        return output

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # This is required for memory-efficient tokenization of large files that we cannot directly load into memory.
        for text in iterable:
            yield from self.encode(text=text)

    def decode(self, ids: list[int]) -> str:
        output = bytes()
        for id in ids:
            output += self.vocab[id]
        return output.decode('utf-8', errors="replace")

        
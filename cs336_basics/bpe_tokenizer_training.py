import regex as re 
from collections import Counter
from tqdm import tqdm
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" # GPT-2 regex 

# Firstly, let's read the file and form a huge string 
def read_file(input_path: str) -> str:
    with open(input_path, 'r') as file:
        text = file.read()
    return text

def strip_special_tokens(input_text: str, special_tokens: list[str]) -> list[str]:
    # If special_tokens is an empty list, don't split at all
    if len(special_tokens) == 0:
        return [input_text]
    else:
        pattern = "|".join(re.escape(t) for t in special_tokens)
        stripped_text = re.split(pattern=pattern, string=input_text)
        return stripped_text
    
def initialize_vocab() -> dict[int, bytes]:
    vocabulary = {}
    for i in range(256):
        value = bytes([i])
        vocabulary[i] = value
    return vocabulary

def pretokenization(input_text: list[str]) -> list[str]:
    output_text = []
    for text in tqdm(input_text, desc = 'Running pretokenization'):
        pre_token_text = re.findall(PAT, text)
        output_text += pre_token_text
    return output_text

def form_freq_table(input_text: list[str]) -> Counter:
    frequency_table = Counter()
    for word in tqdm(input_text, desc='Forming the frequency table'):
        key = tuple(bytes([b]) for b in word.encode('utf-8'))
        frequency_table[key] += 1
    return frequency_table

def form_pair_table(frequency_table: Counter) -> Counter:
    pair_table = Counter()
    for key in tqdm(frequency_table, desc='Forming the pair table'):
        for i in range(len(key) - 1):
            pair_key = tuple((key[i], key[i + 1]))
            pair_table[pair_key] += frequency_table[key]
    return pair_table

def find_max_pair(pair_table: Counter) -> tuple:
    max_occur = - 1
    max_pair = None
    for key in pair_table:
        if pair_table[key] > max_occur:
            max_pair = key
            max_occur = pair_table[key]
        elif pair_table[key] == max_occur:  # If there is a tie, then we resolve it by lexicographical order
            if key > max_pair:
                max_pair = key 
    return max_pair

def update_freq_pair_tables(frequency_table: Counter, pair_table: Counter, max_pair: tuple) -> (Counter, Counter):
    new_freq_table = {}
    new_token = max_pair[0] + max_pair[1]

    for key in frequency_table:
        value = frequency_table[key]
        if max_pair[0] not in key or max_pair[1] not in key:
            new_freq_table[key] = value
        else:
            new_key = []
            i = 0
            while i < len(key) - 1:
                if key[i] == max_pair[0] and key[i + 1] == max_pair[1]:
                    new_key.append(new_token)
                    # Here we have to update the pair_table since we merged two tokens/bytes
                    if i - 1 >= 0: 
                        pair_table[(key[i - 1], key[i])] -= value
                        pair_table[(key[i - 1], new_token)] += value

                    if i + 2 <= len(key) - 1:
                        pair_table[(key[i + 1], key[i + 2])] -= value
                        pair_table[(new_token, key[i + 2])] += value
                    i += 2
                else:
                    new_key.append(key[i])
                    i += 1
            if i == len(key) - 1:
                new_key.append(key[i])
            new_key = tuple(new_key)
            new_freq_table[new_key] = value
    pair_table[max_pair] = 0 

    return new_freq_table, pair_table

def tokenization_training(input_path: str, vocab_size: int, special_tokens: list[str]) -> (dict[int, bytes], list[tuple[bytes, bytes]]):
    # Firstly, initialize the vocabulary 
    vocabulary = initialize_vocab()
    counter = 256
    pbar = tqdm(total = vocab_size)

    # Secondly, read the input data, strip special tokens, and pretokenize
    text = read_file(input_path=input_path)
    stripped_text = strip_special_tokens(input_text=text, special_tokens=special_tokens)
    pretoken_text = pretokenization(input_text=stripped_text)

    # Thirdly, we perform BPE merges
    frequency_table = form_freq_table(input_text=pretoken_text)
    pair_table = form_pair_table(frequency_table=frequency_table)
    merges_list = []

    while len(vocabulary) < vocab_size - len(special_tokens):
        max_pair = find_max_pair(pair_table=pair_table)
        frequency_table, pair_table = update_freq_pair_tables(frequency_table=frequency_table, pair_table=pair_table, max_pair=max_pair)

        merges_list.append(max_pair)
        vocabulary[counter] = max_pair[0] + max_pair[1]
        counter += 1
        pbar.update(1)

    # Lastly, we add the special tokens
    for i in range(len(special_tokens)):
        vocabulary[counter + i] = special_tokens[i].encode('utf-8')
        pbar.update(1)
    pbar.close()

    return vocabulary, merges_list







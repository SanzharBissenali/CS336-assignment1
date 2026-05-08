# pip install datasets
import json
from datasets import load_dataset
from cs336_basics.bpe_tokenizer_training import tokenization_training

# Firstly, we load the dataset and store it as a text file
dataset = load_dataset("roneneldan/TinyStories")

text = '<|endoftext|>'.join(dataset['train']['text'])
# for i in tqdm(range(len(dataset['train']['text'])), desc='Reading from the dataset'):
#     text += dataset['train']['text'][i]
#     text += '<|endoftext|>'
print(f"The total number of characters in the training set is {len(text)}")

with open('output/TinyStories.txt', 'w') as file:
    file.write(text)

# Secondly, we do the BPE training, as we have all the ingredients: 1) filepath, 2) vocab_size, 3) special tokens
vocabulary, merge_list = tokenization_training(input_path='output/TinyStories.txt', vocab_size=10000, special_tokens=['<|endoftext|>'])

# Thirdly, we store these as json and txt files for later retrieval
# JSON can't have integers as keys, so we invert the vocabulary mapping 
# Moreover, JSOn cannot serialize bytes, so we store them as strings.

inverse_vocab = {}
for int_id in vocabulary:
    byte_key = vocabulary[int_id].decode('latin-1')
    inverse_vocab[byte_key] = int_id

with open('output/vocab_tiny_stories.json', 'w') as file:
    json.dump(inverse_vocab, file)

with open('output/merges_list.txt', 'w') as file:
    for i in range(len(merge_list)):    # merge list is list[tuple[bytes, bytes]]
        first_token = merge_list[i][0].decode('latin-1')
        second_token = merge_list[i][1].decode('latin-1')
        file.write(first_token + " " + second_token + '\n')

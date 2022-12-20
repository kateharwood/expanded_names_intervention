from gensim.models import Word2Vec
import jsonlines
import re
import string
import random
import numpy as np
import argparse
from scipy.special import softmax
import math

parser = argparse.ArgumentParser()
parser.add_argument('--num_female_names', type=int, required=True)
parser.add_argument('--num_male_names', type=int, required=True)
parser.add_argument('--dest', type=str, required=True)
parser.add_argument('--less_target', type=bool, default=False)
parser.add_argument('--name_matching_probabilities', type=str, required=True)
args = parser.parse_args()

if args.less_target:
     with open('data/target_pairs_less.txt', 'r') as f:
        lines = f.read().splitlines()
else: 
    with open('data/target_pairs.txt', 'r') as f:
        lines = f.read().splitlines()
lines = [line.strip() for line in lines]
target_pairs = [line.split(" - ") for line in lines]

with open('data/female_names.txt', 'r') as f:
    lines = f.read().splitlines()
    lines = [line.lower() for line in lines]
female_names = lines[:args.num_female_names]
with open('data/male_names.txt', 'r') as f:
    lines = f.read().splitlines()
    lines = [line.lower() for line in lines]
male_names = lines[:args.num_male_names]

name_matching_probabilities = np.load(args.name_matching_probabilities)
print(name_matching_probabilities.shape)

articles = []
corpus = ['data/parsed_wiki_articles_tokenized_partial7.jsonl',
    'data/parsed_wiki_articles_tokenized_partial6.jsonl',
    'data/parsed_wiki_articles_tokenized_partial2.jsonl', 
    'data/parsed_wiki_articles_tokenized_partial3.jsonl',
    'data/parsed_wiki_articles_tokenized_partial4.jsonl', 
    'data/parsed_wiki_articles_tokenized_partial5.jsonl']
for filename in corpus:
    print(filename)
    for article in jsonlines.open(filename):
        # 50/50 replace the words in the sentence with the opposite target words / names
        replace = random.randint(0,1)
        if replace:
            # switch target gendered words
            for token in article:
                for pair in target_pairs:
                    if token == pair[0]:
                        article[article.index(token)] = pair[1]
                    elif token == pair[1]:
                        article[article.index(token)] = pair[0]
            # switch name
            for token in article:
                if token in female_names:
                    name_index = female_names.index(token)
                    index_of_female_name = name_index*len(male_names)
                    matching_probabilities = name_matching_probabilities[index_of_female_name : index_of_female_name + len(male_names)]
                    assert(len(matching_probabilities) == len(male_names))
                    assert(math.isclose(np.sum(matching_probabilities), len(male_names)/len(female_names)))
                    if np.sum(matching_probabilities) != 1: # when there are more male names than female names
                        matching_probabilities = softmax(matching_probabilities)
                    idx = np.random.choice(list(range(len(matching_probabilities))), p=matching_probabilities)
                    name_swap = male_names[idx % len(male_names)]
                    article[article.index(token)] = name_swap

                if token in male_names:
                    name_index = male_names.index(token)
                    matching_probabilities = name_matching_probabilities[name_index:len(name_matching_probabilities):len(male_names)]
                    assert(len(matching_probabilities) == len(female_names))
                    idx = np.random.choice(list(range(len(matching_probabilities))), p=matching_probabilities)
                    name_swap = female_names[idx%len(male_names)]
                    article[article.index(token)] = name_swap
        articles.append(article)
print('Done replacing articles.')

# CBOW model, 300 length vector, epochs at default 5 all as in the original paper
# leaving window as default 5 bc it is not mentioned in paper
model = Word2Vec(articles, vector_size = 300, min_count = 10) 
model.save(args.dest)

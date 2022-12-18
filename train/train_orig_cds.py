from gensim.models import Word2Vec
import jsonlines
import re
import string
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dest', type=str, required=True)
parser.add_argument('--less_target', type=bool, default=False)
args = parser.parse_args()

if args.less_target:
     with open('data/target_pairs_less.txt', 'r') as f:
        lines = f.read().splitlines()
else: 
    with open('data/target_pairs.txt', 'r') as f:
        lines = f.read().splitlines()

lines = [line.strip() for line in lines]

target_pairs = [line.split(" - ") for line in lines]

articles = []
corpus = ['data/parsed_wiki_articles_tokenized_partial6.jsonl',
    'data/parsed_wiki_articles_tokenized_partial7.jsonl',
    'data/parsed_wiki_articles_tokenized_partial2.jsonl', 
    'data/parsed_wiki_articles_tokenized_partial3.jsonl',
    'data/parsed_wiki_articles_tokenized_partial4.jsonl', 
    'data/parsed_wiki_articles_tokenized_partial5.jsonl']
for filename in corpus:
    print(filename)
    for article in jsonlines.open(filename):
        # 50/50 replace the words in the sentence with the opposite target words
        replace = random.randint(0,1)
        if replace:
            for pair in target_pairs:
                for token in article:
                    if token == pair[0]:
                        article[article.index(token)] = pair[1]
                    elif token == pair[1]:
                        article[article.index(token)] = pair[0]
        articles.append(article)
print('Done replacing articles')

# CBOW model, 300 length vector, epochs at default 5 all as in the original paper
# leaving window as default 5 bc it is not mentioned in paper
model = Word2Vec(articles, vector_size = 300, min_count = 10) 
model.save(args.dest)

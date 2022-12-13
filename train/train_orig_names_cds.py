from gensim.models import Word2Vec
import jsonlines
import re
import string
import random

with open('data/target_pairs.txt', 'r') as f:
    lines = f.read().splitlines()
lines = [line.strip() for line in lines]
target_pairs = [line.split(" - ") for line in lines]

with open('data/bipartite_name_matches.txt', 'r') as f:
    lines = f.read().splitlines()
lines = [line.strip() for line in lines]
name_pairs = [line.split(" ") for line in lines]

articles = []
corpus = ['data/parsed_wiki_articles_tokenized_partial2.jsonl', 
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
            for token in article: # need to go through again with switched tokens (we have names "queen" and "man")
                for name_pair in name_pairs:
                    if token == name_pair[0]:
                        article[article.index(token)] = name_pair[1]
                    elif token == name_pair[1]:
                        article[article.index(token)] = name_pair[0]
        articles.append(article)
print('Done replacing articles.')

# CBOW model, 300 length vector, epochs at default 5 all as in the original paper
# leaving window as default 5 bc it is not mentioned in paper
model = Word2Vec(articles, vector_size = 300, min_count = 10) 
model.save('models/orig_names_cds.kv')

from gensim.models import Word2Vec
import jsonlines
import re
import string

articles = []
for article in jsonlines.open('data/parsed_wiki_articles_tokenized_partial2.jsonl'):
    articles.append(article)
for article in jsonlines.open('data/parsed_wiki_articles_tokenized_partial3.jsonl'):
    articles.append(article)
for article in jsonlines.open('data/parsed_wiki_articles_tokenized_partial4.jsonl'):
    articles.append(article)
for article in jsonlines.open('data/parsed_wiki_articles_tokenized_partial5.jsonl'):
    articles.append(article)
for article in jsonlines.open('data/parsed_wiki_articles_tokenized_partial6.jsonl'):
    articles.append(article)
for article in jsonlines.open('data/parsed_wiki_articles_tokenized_partial7.jsonl'):
    articles.append(article)


print(len(articles))

# CBOW model, 300 length vector, epochs at default 5 all as in the original paper
# leaving window as default 5 bc it is not mentioned in paper
model = Word2Vec(articles, vector_size = 300, min_count=10) 

model.save('models_final/baseline.kv')

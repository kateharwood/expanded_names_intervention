import os
import json
import jsonlines
import string
import re
from gensim.models import Word2Vec
from nltk.parse import CoreNLPParser


nlp = CoreNLPParser(url='http://localhost:8000')

def parse_wikipedia():
    wiki_articles = []
    stop_early = 3000000
    i = 0
    for filename in os.listdir('../end2end/wiki-pages'):
        wiki_file = os.path.join('../end2end/wiki-pages', filename)
        with open(wiki_file, 'r') as f:
            lines = [json.loads(jline) for jline in f.read().splitlines()]
        for line in lines:
            i += 1
            if i % 100 == 0:
                print(i)
            if i % 10000 == 0:
                with jsonlines.open('help.jsonl', mode='a') as writer:
                    writer.write_all([i])
                writer.close()
                with jsonlines.open('wiki_articles_tokenized_partial7.jsonl', mode='a') as writer:
                    writer.write_all(wiki_articles)
                writer.close()
                wiki_articles = []
            if i > stop_early:
                break
            if i >= 2614500:
                wiki_articles.append(list(nlp.tokenize(line['text'])))

def lowercase_and_punctuation():
    remove = string.punctuation
    remove = remove.replace("-", "") 
    remove = remove.replace("_", "")
    pattern = r"[{}]".format(remove)

    txt = ")*^%{}[]thi's - is - @@#!a !%%!!%- test."
    re.sub(pattern, "", txt) 

    # corpus = ['wiki_articles_tokenized_partial.jsonl', 'wiki_articles_tokenized_partial2.jsonl', 
    #     'wiki_articles_tokenized_partial3.jsonl', 'wiki_articles_tokenized_partial4.jsonl', 'wiki_articles_tokenized_partial5.jsonl']
    # corpus = ['wiki_articles_tokenized_partial6.jsonl']
    corpus = ['wiki_articles_tokenized_partial7.jsonl']
    for filename in corpus:
        articles = []
        for article in jsonlines.open(filename):
            lowercased = [re.sub(pattern, "", token.lower()) for token in article]
            lowercased = [token for token in lowercased if token] # remove empty strings
            articles.append(lowercased)

        with jsonlines.open("parsed_" + filename, mode='a') as writer:
            writer.write_all(articles)
        writer.close()

# parse_wikipedia()
lowercase_and_punctuation()
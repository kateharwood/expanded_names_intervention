from gensim.models import Word2Vec
import jsonlines
import re
import string


# Set up remove punctuation
# remove = string.punctuation
# remove = remove.replace("-", "") 
# remove = remove.replace("_", "")
# pattern = r"[{}]".format(remove)

# txt = ")*^%{}[]thi's - is - @@#!a !%%!!%- test."
# re.sub(pattern, "", txt) 

articles = []
for article in jsonlines.open('parsed_wiki_articles_tokenized_partial2.jsonl'):
    # lowercased = [re.sub(pattern, "", token.lower()) for token in article]
    # lowercased = [token for token in lowercased if token] # remove empty strings
    articles.append(article)
for article in jsonlines.open('parsed_wiki_articles_tokenized_partial3.jsonl'):
    # lowercased = [re.sub(pattern, "", token.lower()) for token in article]
    # lowercased = [token for token in lowercased if token] 
    articles.append(article)
for article in jsonlines.open('parsed_wiki_articles_tokenized_partial4.jsonl'):
    # lowercased = [re.sub(pattern, "", token.lower()) for token in article]
    # lowercased = [token for token in lowercased if token] 
    articles.append(article)
for article in jsonlines.open('parsed_wiki_articles_tokenized_partial5.jsonl'):
    # lowercased = [re.sub(pattern, "", token.lower()) for token in article]
    # lowercased = [token for token in lowercased if token]
    articles.append(article)

# print(articles)
print(len(articles))

# CBOW model, 300 length vector, epochs at default 5 all as in the original paper
# leaving window as default 5 bc it is not mentioned in paper
model = Word2Vec(articles, vector_size = 300, min_count=10) 

model.save('baseline.kv')

from gensim.models import KeyedVectors
import numpy as np
import sklearn.manifold
from matplotlib import pyplot as plt


def get_most_gendered_words():
    with open('gendered_pairs.txt', 'r') as f:
        lines = f.read().splitlines()

    pairs = [line.split(" ") for line in lines]
    num_pairs = len(pairs)

    baseline = KeyedVectors.load('baseline.kv')
    cov_matrix_sum = np.zeros((300,300))
    for pair in pairs:
        word1 = baseline.wv[pair[0]]
        word2 = baseline.wv[pair[1]]
        pair = np.row_stack((word1,word2))
        mean = np.mean(pair, axis=0)
        print(pair[0] - mean)
        cov_matrix = np.multiply(pair[0] - mean, (np.expand_dims(pair[0]-mean, axis=0).T))
        print(cov_matrix.shape)
        cov_matrix_sum = np.add(cov_matrix_sum, cov_matrix)
        print(cov_matrix_sum.shape)

    eigenvals, eigenvectors = np.linalg.eig(cov_matrix_sum)
    gender_vector = eigenvectors[:, 0]
    print(gender_vector.shape)

    print(baseline.wv.most_similar(positive=[gender_vector], topn=100))
    print(baseline.wv.most_similar(negative=[gender_vector], topn=100))

    # cds_model = KeyedVectors.load('orig_cds.kv')

    # print(cds_model.wv.most_similar(positive=[gender_vector], topn=100))
    # print(cds_model.wv.most_similar(negative=[gender_vector], topn=100))

    most_similar_words_female = baseline.wv.most_similar(positive=[gender_vector], topn=500)
    most_similar_words_male = baseline.wv.most_similar(negative=[gender_vector], topn=500)
    most_similar_words = most_similar_words_female + most_similar_words_male
    return most_similar_words, baseline

def cluster_baseline():
    most_similar_words, baseline = get_most_gendered_words()
    embeddings = np.array([baseline.wv[word[0]] for word in most_similar_words])

    tsne = sklearn.manifold.TSNE()
    tsne_results = tsne.fit_transform(embeddings)
    print(tsne_results)
    print(tsne_results.shape)
    x = embeddings[:, 0]
    y = embeddings[:, 1]
    x_female = x[:500]
    x_male = x[500:]
    y_female = y[:500]
    y_male = y[500:]

    print(x)
    print(x_female)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(x_female, y_female, s=10, c='b', marker="s", label='female')
    ax1.scatter(x_male,y_male, s=10, c='r', marker="o", label='male')
    plt.legend(loc='upper left')
    plt.savefig('figs/baseline.png')

def cluster_orig_cds():
    cds_model = KeyedVectors.load('orig_cds.kv')
    most_similar_words, baseline = get_most_gendered_words()
    embeddings = np.array([cds_model.wv[word[0]] for word in most_similar_words])

    tsne = sklearn.manifold.TSNE(perplexity=60)
    tsne_results = tsne.fit_transform(embeddings)
    print(tsne_results)
    print(tsne_results.shape)
    x = embeddings[:, 0]
    y = embeddings[:, 1]
    x_female = x[:500]
    x_male = x[500:]
    y_female = y[:500]
    y_male = y[500:]

    print(x)
    print(x_female)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(x_female, y_female, s=10, c='b', marker="s", label='female')
    ax1.scatter(x_male,y_male, s=10, c='r', marker="o", label='male')
    plt.legend(loc='upper left')
    plt.savefig('figs/orig_cds.png')


cluster_baseline()
# cluster_orig_cds()
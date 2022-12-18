from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import numpy as np
import sklearn.manifold
from matplotlib import pyplot as plt
from sklearn.svm import SVC
import random

def get_most_gendered_words(num_gendered_words=500):
    with open('analysis/family_pairs.txt', 'r') as f:
        lines = f.read().splitlines()

    pairs = [line.split(" ") for line in lines]
    num_pairs = len(pairs)

    baseline = KeyedVectors.load('models/baseline.kv')
    cov_matrix_sum = np.zeros((300,300))
    for pair in pairs:
        word1 = baseline.wv[pair[0]]
        word2 = baseline.wv[pair[1]]
        pair = np.row_stack((word1,word2))
        mean = np.mean(pair, axis=0)

        male_cov_matrix = np.multiply(pair[0] - mean, (np.expand_dims(pair[0]-mean, axis=0).T))
        female_cov_matrix = np.multiply(pair[1] - mean, (np.expand_dims(pair[1]-mean, axis=0).T))
        cov_matrix = np.add(male_cov_matrix, female_cov_matrix)
        cov_matrix_sum = np.add(cov_matrix_sum, cov_matrix)

    eigenvals, eigenvectors = np.linalg.eig(cov_matrix_sum)
    gender_vector = eigenvectors[:, 0]

    most_similar_words_female = baseline.wv.most_similar(positive=[gender_vector], topn=5000) # just get a bunch
    most_similar_words_male = baseline.wv.most_similar(negative=[gender_vector], topn=5000)
    
    # if we want to not include names in the top gendered words
    # with open('data/female_names.txt', 'r') as f:
    #     female_names = f.read().splitlines()
    #     female_names = [name.lower() for name in female_names]
    # with open('data/male_names.txt', 'r') as f:
    #     male_names = f.read().splitlines()
    #     male_names = [name.lower() for name in male_names]
    # most_similar_words_female = [word for word in most_similar_words_female if word[0] not in female_names]
    # most_similar_words_male = [word for word in most_similar_words_male if word[0] not in male_names]

    # print(len(most_similar_words_female))
    # print(len(most_similar_words_male))
    
    assert(len(most_similar_words_female) >= num_gendered_words)
    assert(len(most_similar_words_male) >= num_gendered_words)
    most_similar_words_female = most_similar_words_female[:num_gendered_words]
    most_similar_words_male = most_similar_words_male[:num_gendered_words]
    most_similar_words = most_similar_words_female + most_similar_words_male
    return most_similar_words, baseline

def cluster(model_to_test='baseline'):
    most_similar_words, baseline = get_most_gendered_words()
    if model_to_test == 'baseline':
        model = baseline
    if model_to_test == 'cds':
        model = KeyedVectors.load('models/orig_cds.kv')    
    if model_to_test == 'names_cds':
        model = KeyedVectors.load('models/orig_names_cds.kv')    
    if model_to_test == 'names_cds_2000':
        model = KeyedVectors.load('models/orig_names_cds_2000.kv')    
    if model_to_test == 'expanded_names_cds':
        model = KeyedVectors.load('models/expanded_names_cds.kv')
    if model_to_test == 'expanded_names_cds_unbalanced_boys_girls':
        model = KeyedVectors.load('models/expanded_names_cds_unbalanced_1500_2500.kv')
    if model_to_test == 'expanded_names_cds_unbalanced_girls_boys':
        model = KeyedVectors.load('models/expanded_names_cds_unbalanced_2500_1500.kv')
    

    #TODO this means there could be different words for each model analysis which isn't good
    # same with the prediction words
    # but I think it's only like 1 for the names and 5 for the expanded names...
    # TODO I think we should take the words out if they aren't in any of the models
    embeddings = []
    for word in most_similar_words:
        if word[0] in model.wv:
            embeddings.append(model.wv[word[0]])
    embeddings = np.asarray(embeddings)
    print(len(embeddings))

    tsne = sklearn.manifold.TSNE()
    tsne_results = tsne.fit_transform(embeddings)
    x = tsne_results[:, 0]
    y = tsne_results[:, 1]
    x_female = x[:500]
    x_male = x[500:]
    y_female = y[:500]
    y_male = y[500:]

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(x_female, y_female, s=10, c='b', marker="s", label='female')
    ax1.scatter(x_male,y_male, s=10, c='r', marker="o", label='male')
    plt.legend(loc='upper left')
    plt.savefig('figs/' + model_to_test + '.png')

def gender_predictor(model_to_test='baseline'):
    most_gendered_words, baseline = get_most_gendered_words(2500) #5000 most gendered words
    female_words = most_gendered_words[:2500]
    male_words = most_gendered_words[2500:]

    if model_to_test == 'baseline':
        model = baseline
    if model_to_test == 'cds':
        model = KeyedVectors.load('models_final/orig_cds_less.kv')    
    if model_to_test == 'names_cds':
        model = KeyedVectors.load('models_final/orig_names_cds.kv')    
    if model_to_test == 'names_cds_2000':
        model = KeyedVectors.load('models_final/orig_names_cds_2000.kv')    
    if model_to_test == 'expanded_names_cds':
        model = KeyedVectors.load('models_final/expanded_names_cds_unbalanced_2000_2000.kv')
    if model_to_test == 'expanded_names_cds_unbalanced_boys_girls':
        model = KeyedVectors.load('models_final/expanded_names_cds_unbalanced_1500_2500.kv')
    if model_to_test == 'expanded_names_cds_unbalanced_girls_boys':
        model = KeyedVectors.load('models_final/expanded_names_cds_unbalanced_2500_1500.kv')

    if model_to_test == 'cds_less':
        model = KeyedVectors.load('models_final/orig_cds_less.kv')    
    if model_to_test == 'names_cds_less':
        model = KeyedVectors.load('models_final/orig_names_cds_less.kv')    
    if model_to_test == 'names_cds_2000_less':
        model = KeyedVectors.load('models_final/orig_names_cds_2000_less.kv')    
    if model_to_test == 'expanded_names_cds_less':
        model = KeyedVectors.load('models_final/expanded_names_cds_unbalanced_2000_2000_less.kv')
    if model_to_test == 'expanded_names_cds_unbalanced_boys_girls_less':
        model = KeyedVectors.load('models_final/expanded_names_cds_unbalanced_1500_2500_less.kv')
    if model_to_test == 'expanded_names_cds_unbalanced_girls_boys_less':
        model = KeyedVectors.load('models_final/expanded_names_cds_unbalanced_2500_1500_less.kv')


    # TODO this is just taking out some words per model, so is not a fair comparison! like above
    female_embedded_words = []
    for word in female_words:
        if word[0] in model.wv:
            female_embedded_words.append([model.wv[word[0]], 1])
    male_embedded_words = []
    for word in male_words:
        if word[0] in model.wv:
            male_embedded_words.append([model.wv[word[0]], 0])
    female_words = female_embedded_words
    male_words = male_embedded_words
    print(len(female_words))
    print(len(male_words))


    female_train = random.sample(list(enumerate(female_words)), k=500)
    female_train_idx, female_train = zip(*female_train)
    male_train = random.sample(list(enumerate(male_words)), k=500)
    male_train_idx, male_train  = zip(*male_train)
    
    train = list(female_train) + list(male_train)
    np.random.shuffle(train)
    assert(len(train) == 1000)

    train = [list(t) for t in zip(*train)]
    clf = sklearn.svm.SVC(kernel='rbf')
    clf.fit(train[0], train[1])
    
    female_test = []
    male_test = []
    for i in range(len(female_words)):
        if i not in list(female_train_idx):
            female_test.append(female_words[i])
    for i in range(len(male_words)):
        if i not in list(male_train_idx):
            male_test.append(male_words[i])

    test = female_test + male_test
    test = [list(t) for t in zip(*test)]
    assert(len(test[0]) <= 4000)
    predictions = clf.predict(test[0])
    print("Accuracy on " + model_to_test + " model words:")
    print(sklearn.metrics.accuracy_score(test[1], predictions))


def word_similarity(model_to_test='baseline'):
    if model_to_test == 'baseline':
        model = KeyedVectors.load('models/baseline.kv') # TODO make this models_final
    if model_to_test == 'cds':
        model = KeyedVectors.load('models_final/orig_cds_less.kv')    
    if model_to_test == 'names_cds':
        model = KeyedVectors.load('models_final/orig_names_cds.kv')    
    if model_to_test == 'names_cds_2000':
        model = KeyedVectors.load('models_final/orig_names_cds_2000.kv')    
    if model_to_test == 'expanded_names_cds':
        model = KeyedVectors.load('models_final/expanded_names_cds_unbalanced_2000_2000.kv')
    if model_to_test == 'expanded_names_cds_unbalanced_boys_girls':
        model = KeyedVectors.load('models_final/expanded_names_cds_unbalanced_1500_2500.kv')
    if model_to_test == 'expanded_names_cds_unbalanced_girls_boys':
        model = KeyedVectors.load('models_final/expanded_names_cds_unbalanced_2500_1500.kv')

    if model_to_test == 'cds_less':
        model = KeyedVectors.load('models_final/orig_cds_less.kv')    
    if model_to_test == 'names_cds_less':
        model = KeyedVectors.load('models_final/orig_names_cds_less.kv')    
    if model_to_test == 'names_cds_2000_less':
        model = KeyedVectors.load('models_final/orig_names_cds_2000_less.kv')    
    if model_to_test == 'expanded_names_cds_less':
        model = KeyedVectors.load('models_final/expanded_names_cds_unbalanced_2000_2000_less.kv')
    if model_to_test == 'expanded_names_cds_unbalanced_boys_girls_less':
        model = KeyedVectors.load('models_final/expanded_names_cds_unbalanced_1500_2500_less.kv')
    if model_to_test == 'expanded_names_cds_unbalanced_girls_boys_less':
        model = KeyedVectors.load('models_final/expanded_names_cds_unbalanced_2500_1500_less.kv')

    similarities = model.wv.evaluate_word_pairs(datapath('simlex999.txt'))
    print(model_to_test)
    print(similarities)


# get_most_gendered_words()
# cluster()
# cluster('cds')
# cluster('names_cds')
# cluster('names_cds_2000')
# cluster('expanded_names_cds')
# cluster('expanded_names_cds_unbalanced_boys_girls')
# cluster('expanded_names_cds_unbalanced_girls_boys')

# gender_predictor()
# gender_predictor('cds')
# gender_predictor('names_cds')
# gender_predictor('names_cds_2000')
gender_predictor('expanded_names_cds')
gender_predictor('expanded_names_cds_unbalanced_boys_girls')
gender_predictor('expanded_names_cds_unbalanced_girls_boys')

# gender_predictor('cds_less')
# gender_predictor('names_cds_less')
# gender_predictor('names_cds_2000_less')
gender_predictor('expanded_names_cds_less')
gender_predictor('expanded_names_cds_unbalanced_boys_girls_less')
gender_predictor('expanded_names_cds_unbalanced_girls_boys_less')


# word_similarity()
# word_similarity('cds')
# word_similarity('names_cds')
# word_similarity('names_cds_2000')
# word_similarity('expanded_names_cds')
# word_similarity('expanded_names_cds_unbalanced_boys_girls')
# word_similarity('expanded_names_cds_unbalanced_girls_boys')

# # word_similarity('cds_less')
# # word_similarity('names_cds_less')
# # word_similarity('names_cds_2000_less')
# word_similarity('expanded_names_cds_less')
# word_similarity('expanded_names_cds_unbalanced_boys_girls_less')
# word_similarity('expanded_names_cds_unbalanced_girls_boys_less')
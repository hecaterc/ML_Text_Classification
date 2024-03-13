# Importing necessary libraries
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from gensim.models import KeyedVectors
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
import gensim.downloader as api

import numpy as np

# Categories
categories = [
    'alt.atheism',
    'talk.religion.misc',
]

# Load dataset
data = fetch_20newsgroups(subset='train', categories=categories)

# Load pre-trained Word2Vec model
#pretrainedpath = "GoogleNews-vectors-negative300.bin"
#w2v_model = KeyedVectors.load_word2vec_format(pretrainedpath, binary=True)
w2v_model = api.load('word2vec-google-news-300')

# Define algorithms
algorithms = {
    'Multinomial Naïve Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Support Vector Machines': SVC(),
    'Decision Trees': DecisionTreeClassifier()
}

# Define feature extractors
feature_extractors = {
    'CountVectorizer': CountVectorizer(),
    'Word2Vec': w2v_model
}

# Benchmarking
results = {}
for algo_name, algo in algorithms.items():
    for extractor_name, extractor in feature_extractors.items():
        if isinstance(extractor, CountVectorizer):
            X = extractor.fit_transform(data.data)
        elif isinstance(extractor, KeyedVectors):
            X = np.array(
                [np.mean([extractor[word] for word in text.split() if word in extractor] or [np.zeros(300)], axis=0) for
                 text in data.data])

        pipeline = Pipeline([
            ('classifier', algo)
        ])

        parameters = {}
        if algo_name == 'Multinomial Naïve Bayes':
            if extractor_name == 'CountVectorizer':
                parameters = {'classifier__alpha': (0.5, 1.0)}
            else:
                continue
        elif algo_name == 'Logistic Regression':
            if extractor_name == 'CountVectorizer':
                parameters = {'classifier__C': (0.1, 1.0, 10.0)}
            else:
                parameters = {'classifier__C': (0.1, 1.0, 10.0)}
        elif algo_name == 'Support Vector Machines':
            if extractor_name == 'CountVectorizer':
                parameters = {'classifier__C': (0.1, 1.0, 10.0), 'classifier__kernel': ('linear', 'rbf')}
            else:
                parameters = {'classifier__C': (0.1, 1.0, 10.0), 'classifier__kernel': ('linear', 'rbf')}
        elif algo_name == 'Decision Trees':
            if extractor_name == 'CountVectorizer':
                parameters = {'classifier__max_depth': (10, 50, 100)}
            else:
                parameters = {'classifier__max_depth': (10, 50, 100)}

        grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X, data.target)

        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        results[(algo_name, extractor_name)] = {'Best Params': best_params, 'Best Score': best_score}

# Find the best configuration
best_config = max(results, key=lambda k: results[k]['Best Score'])

# Print and save results to a file
with open('Ryan_Chan_Task0_Text_Classification.txt', 'w') as f:
    f.write("Algorithm                Feature Extractor    Best Parameters                      Best Score\n")
    for config, result in results.items():
        algo_name, extractor_name = config
        best_params = str(result['Best Params'])
        best_score = str(result['Best Score'])
        # Calculate spacing for alignment
        algo_spacing = max(0, 30 - len(algo_name))
        extractor_spacing = max(0, 20 - len(extractor_name))
        params_spacing = max(0, 40 - len(best_params))
        # Write to file with proper spacing
        f.write(f"{algo_name}{' ' * algo_spacing}{extractor_name}{' ' * extractor_spacing}{best_params}{' ' * params_spacing}{best_score}\n")


print("Best Configuration:")
print("Algorithm:", best_config[0])
print("Feature Extractor:", best_config[1])
print("Best Parameters:", results[best_config]['Best Params'])
print("Best Score:", results[best_config]['Best Score'])

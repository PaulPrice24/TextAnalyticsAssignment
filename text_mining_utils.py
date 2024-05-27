"""
Created on Wed Feb  7 11:42:55 2024

@author: aurelia power
"""

import re, pandas as pd, numpy as np, matplotlib.pyplot as plt
import nltk
from collections import Counter 
import warnings 
warnings.filterwarnings('ignore') 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics import classification_report 
from sklearn.model_selection import cross_val_predict 
import wordcloud 

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from pandas.plotting import parallel_coordinates

from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import RFECV


#############################################
############ text visualisations ############

"""
takes in a list of tagged documents and a POS(as a string) 
returns the normalised count of a POS for each tagged document 
"""
def normalisePOSCounts(tagged_docs, pos):
    counts = [] 
    for doc in tagged_docs:
        count = 0 
        for pair in doc:
            if pair[1] == pos:
                count += 1 
        counts.append(count) 
    lengths = [len(doc) for doc in tagged_docs] 
    return [count/length for count, length in zip(counts, lengths)] 

"""
takes in a list of documents, a POS(as a string), and a list of categories/labels 
it tags the documents and calls the above function 
it then plots the normalised frequency of the POS across all labels 
"""
def plotPOSFreq(docs, pos, labels):
    tagged_docs = [nltk.pos_tag(nltk.word_tokenize(doc)) for doc in docs] 
    normalised_counts = normalisePOSCounts(tagged_docs, pos) 
    plt.bar(np.arange(len(docs)), normalised_counts, align='center') 
    plt.xticks(np.arange(len(docs)), labels, rotation=40) 
    plt.xlabel('Label (Category)') 
    plt.ylabel(pos + ' frequency') 
    plt.title('Frequency distribution of ' + pos) 

## function to generate the word cloud for a given topic/class
def generate_cloud(text, topic, bg_colour='black', min_font=10):
    cloud = wordcloud.WordCloud(width=700, height=700, random_state=1, background_color=bg_colour, min_font_size=min_font).generate(text) 
    plt.figure(figsize=(7, 7), facecolor=None) 
    plt.imshow(cloud) 
    ##plt.axis('off') 
    plt.tight_layout(pad=0) 
    plt.xlabel(topic) 
    plt.xticks([]) 
    plt.yticks([]) 

## function to generate multiple word clouds for a set of topics/classes/categories
def generate_wordclouds(texts, categories, bg_colour, min_font=10):
  fig = plt.figure(figsize=(21, 7))
  for i in range(len(texts)):
    ax = fig.add_subplot(1,3,i+1)
    cloud = wordcloud.WordCloud(width=700, height=700, random_state=1, 
                      background_color=bg_colour, 
                      min_font_size=min_font).generate(texts[i])
    ax.imshow(cloud)
    ax.axis('off')
    ax.set_title(categories[i])

#########################################################
########## vectorisation functions ######################
"""=minFont
NOTE: all vectorisers from sklearn discard punctuation, which may not be appropriate.
So, I have specified a regex to deal with this situation.
"""
token_regex = r"\w+(?:'\w+)?|[^\w\s]" 

"""
takes in a list of documents, applies the CountVectoriser from sklearn
using the following params by default: decode_error='replace', strip_accents=None, 
lowercase=False, ngram_range=(1, 1)  then it builds and returns a data frame 
"""
def build_count_matrix(docs, decode_error='replace', strip_accents=None, lowercase=False, token_pattern=token_regex, ngram_range=(1, 1)):    
    vectorizer = CountVectorizer(decode_error=decode_error, strip_accents=strip_accents, lowercase=lowercase, token_pattern=token_pattern, ngram_range=ngram_range) 
    X = vectorizer.fit_transform(docs) 
    terms = list(vectorizer.get_feature_names_out()) 
    count_matrix = pd.DataFrame(X.toarray(), columns=terms) 
    return count_matrix.fillna(0) 

## function to generate a matrix with normalised frequencies  same params as above
def build_tf_matrix(docs, decode_error='replace', strip_accents=None, lowercase=False, token_pattern=token_regex, ngram_range=(1, 1)):
    count_matrix = build_count_matrix(docs, decode_error=decode_error, strip_accents=strip_accents, lowercase=lowercase, token_pattern=token_pattern, ngram_range=ngram_range) 
    doc_lengths = count_matrix.sum(axis=1) 
    return count_matrix.divide(doc_lengths, axis=0) 

## function to generate a matrix with tfidfs scores  same params as above
def build_tfidf_matrix(docs, decode_error='replace', strip_accents=None, lowercase=False, token_pattern=token_regex, ngram_range=(1, 1)):
    vectorizer = TfidfVectorizer(decode_error=decode_error, strip_accents=strip_accents, lowercase=lowercase, token_pattern=token_pattern, ngram_range=ngram_range) 
    X = vectorizer.fit_transform(docs) 
    terms = list(vectorizer.get_feature_names_out()) 
    tfidf_matrix = pd.DataFrame(X.toarray(), columns=terms) 
    return tfidf_matrix.fillna(0) 

#########################################################
############# validation functions ######################

"""function to train and x-validate across acc, rec, prec  
and get the classification report"""
def printClassifReport(clf, X, y, folds=5):
    predictions = cross_val_predict(clf, X, y, cv=folds) 
    print(classification_report(y, predictions)) 


"""function to train and x-validate across acc, rec, prec  
for 3 different matrices and plot them"""
def plot_avg_performance_for_3matrices(clf, clf_name,matrices, matrices_names, y, cv=5):
  from sklearn.metrics import classification_report
  from sklearn.model_selection import cross_val_predict   
  results = [classification_report(cross_val_predict(clf, matrix, y, cv=5), y, output_dict=True) for matrix in matrices]
  labels = ['accuracy', 'precision', 'recall']
  x = np.arange(len(labels)) ; width = 0.25 ; fig, ax = plt.subplots()
  rects1 = ax.bar(x , [results[0]['accuracy'], results[0]['macro avg']['precision'], results[0]['macro avg']['recall']], width, label=matrices_names[0])
  #ax.bar_label(rects1, padding=3)
  rects2 = ax.bar(x + width,  [results[1]['accuracy'], results[1]['macro avg']['precision'], results[1]['macro avg']['recall']], width, label=matrices_names[1])
  #ax.bar_label(rects2, padding=3)
  rects3 = ax.bar(x + width*2, [results[2]['accuracy'], results[2]['macro avg']['precision'], results[2]['macro avg']['recall']], width, label=matrices_names[2])
  #ax.bar_label(rects3, padding=3)
  ax.set_ylabel('Scores') 
  ax.set_title('Cross Validation Scores across 3 different matrices ' + clf_name)
  ax.set_xticks(x + width); ax.set_xticklabels(labels)
  ax.legend(); fig.tight_layout(); plt.show()

"""function to train and x-validate across acc, rec, prec  
and across 3 different classifiers and plot them"""
def plot_avg_performance_across_3clfs(clfs, clf_names, matrix, matrix_name, y, cv=5):
  results = [classification_report(cross_val_predict(clf, matrix, y, cv=5), y, output_dict=True) for clf in clfs]
  labels = ['accuracy', 'precision', 'recall']
  x = np.arange(len(labels)) ; width = 0.25 ; fig, ax = plt.subplots()
  rects1 = ax.bar(x , [results[0]['accuracy'], results[0]['macro avg']['precision'], results[0]['macro avg']['recall']], width, label=clf_names[0])
  #ax.bar_label(rects1, padding=3)
  rects2 = ax.bar(x + width,  [results[1]['accuracy'], results[1]['macro avg']['precision'], results[1]['macro avg']['recall']], width, label=clf_names[1])
  #ax.bar_label(rects2, padding=3)
  rects3 = ax.bar(x + width*2, [results[2]['accuracy'], results[2]['macro avg']['precision'], results[2]['macro avg']['recall']], width, label=clf_names[2])
  #ax.bar_label(rects3, padding=3)
  ax.set_ylabel('Scores') 
  ax.set_title('Cross Validation Scores across 3 different classifiers trained on ' + matrix_name)
  ax.set_xticks(x + width); ax.set_xticklabels(labels)
  ax.legend(); fig.tight_layout(); plt.show()


#########################################################
############# word stats functions ######################
## function to print the n most frequent tokens in a text belonging to a given topic
def print_n_mostFrequent(topic, text, n):
    tokens = nltk.word_tokenize(text) 
    counter = Counter(tokens) 
    n_freq_tokens = counter.most_common(n) 
    print("=== "+ str(n) + " most frequent tokens in "  + topic + " ===") 
    for token in n_freq_tokens:
        print("\tFrequency of", "\"" + token[0] + "\" is:", token[1]/len(tokens)) 
        
## function to find the frequency of a token in several texts belonging to same topics/classes        
def token_percentage(token, texts):
    token_count = 0 
    all_tokens_count = 0 
    for text in texts:
        tokens = nltk.word_tokenize(text) 
        token_count += tokens.count(token) 
        all_tokens_count += len(tokens) 
    return token_count/all_tokens_count * 100 
        
#########################################################
############# preprocessing functions ###################
## function to carry out some initial cleaning
def clean_doc(doc, clean_operations):
    for key, value in clean_operations.items():
        doc = re.sub(key, value, doc) 
    return doc 

## function to resolve contractions
def resolve_contractions(doc, contr_dict):
    for key, value, in contr_dict.items():
        doc = re.sub(key, value, doc) 
    return doc 

## function to carry out concept typing, resolve synonyms and word variations
def improve_bow(doc, repl_dict):
    for key in repl_dict.keys():
        for item in repl_dict[key]:
            doc = re.sub(item, key, doc, flags=re.IGNORECASE)
    return doc

## function to remove tokens using POS  tags
def remove_terms_by_POS(doc, tags_to_remove):
    tagged_doc = nltk.pos_tag(nltk.word_tokenize(doc)) ## (sea, 'NN')
    new_doc = [pair[0] for pair in tagged_doc if pair[1] not in tags_to_remove]
    new_doc = ' '.join(new_doc)
    ## replace space before punctuation sign
    return re.sub(r' (?=[!\.,?:;])', "", new_doc)

## function to lower case at the beginning of the sentence only
def lower_at_begining(doc):
    sents = nltk.sent_tokenize(doc)
    ##tokenised_sents = [nltk.word_tokenize(token) in sent for sent in sents]##
    tokenised_sents = [re.sub(sent[0], sent[0].lower(), sent)
                       for sent in sents]
    return ' '.join(tokenised_sents)

## function to remove stop words and/or punctuation
def remove_sw_punct(doc, to_remove):
    tokens = nltk.word_tokenize(doc)
    return re.sub(r' (?=[!\.,?:;])', "",
                  ' '.join([token for token in tokens if token not in to_remove]))


## function to remove short tokens
def remove_by_token_len(doc, n):
    tokens = nltk.word_tokenize(doc);
    return re.sub(r' (?=[!\.,?:;])', "",
                  ' '.join([token for token in tokens if len(token) > n]))

## function to remove digits
def remove_d(doc):
    return re.sub(r'\d+', '', doc)

## function to carry out stemming
def stem_doc(doc, stemmer):
    tokens = nltk.word_tokenize(doc)
    return ' '.join([stemmer.stem(t) for t in tokens])


########################################################################
############################# clustering ###############################
def k_means_clustering(X, k=2, initialisation='random'):
    model = KMeans(k, init=initialisation, random_state=1)
    model.fit(X)
    cluster_labels = model.labels_
    cluster_centers = model.cluster_centers_
    return (model, cluster_labels, cluster_centers)

def centroids_across_terms(X, centers, labels, title):
    labels = sorted(set(labels))
    centersFrame = pd.DataFrame(centers, index=labels, columns=X.columns)
    centersFrame['cluster'] = labels
    plt.figure(figsize=(10, 5))
    plt.title(title)
    parallel_coordinates(centersFrame, 'cluster', marker='o')
    plt.legend(labels, loc='best')

def agglom_clustering(X, k=2, simMethod='cosine', linkMethod='ward'):
    model= AgglomerativeClustering(n_clusters=k, affinity=simMethod, linkage=linkMethod)
    model.fit(X)
    return (model, model.labels_)

def visualise_dendrogram(X, linkageMethod='single', xlabel='', ylabel=''):
    Z = linkage(X, linkageMethod)
    plt.figure(figsize=(10, 7))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    dendrogram(Z, labels=X.index, leaf_rotation=90)

def plot_clusters(X, labels, title, xlabel, ylabel):
    clustersDF = pd.DataFrame([labels], columns=X.index)
    plt.plot(clustersDF.iloc[0], color='green', marker='x', linewidth=0, markersize=10)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(list(clustersDF.columns))
    plt.yticks(np.arange(len(set(labels))))


########################################################################
######################### feature selection ############################
########################################################################

# a function that will allow us to pass the method for weighting, the method for selection, 
# and the number of features to retain; 
# it also has the option to output the scores of top n features
def stat_univariate_fs(X, y, weight_method, selection_method, num_features, scores_to_print=25):
    X_reduced = GenericUnivariateSelect(score_func=weight_method, mode=selection_method,
					param=num_features).fit(X, y)
    scores = pd.DataFrame(X_reduced.scores_)
    columns = pd.DataFrame(X.columns)
    features_scores = pd.concat([columns, scores], axis=1)
    features_scores.columns = ['Attribute', 'Weight']
    print("Top", scores_to_print, "features:")
    print(features_scores.nlargest(scores_to_print, 'Weight'))
    return X_reduced.transform(X)

# a function that will allow us to use a specific algorithm 
# for weighting and the number of features to retain
# it also has the option to output the scores of top n features
def clf_univariate_fs(X, y, learner, 
                            num_features, scores_to_print=25):
    learner = learner.fit(X, y)
    scores = None
    if 'feature_importances_' in learner.__dir__():
        scores = pd.DataFrame(learner.feature_importances_)
    elif 'coef_' in learner.__dir__():
        scores = pd.DataFrame([np.max(np.abs(x)) for x in learner.coef_.T])
    columns = pd.DataFrame(X.columns)
    features_scores = pd.concat([columns, scores], axis=1)
    features_scores.columns = ['Attribute', 'Weight']
    print("Top", scores_to_print, "features:")
    print(features_scores.nlargest(scores_to_print, 'Weight'))
    return SelectFromModel(learner, prefit=True).transform(X)

# a function that will allow us to use a specific algorithm 
# for weighting and the number of features to retain from sequential selection
def sequential_fs(X, y, learner, num_features, direction='forward'):
	seq_selector = SequentialFeatureSelector(learner, 
			n_features_to_select=num_features, direction=direction);
	X_reduced = seq_selector.fit_transform(X, y);
	print([feature for feature in seq_selector.get_feature_names_out()]);
	return X_reduced;

# a function that will allow us to use a specific algorithm 
# for weighting and the number of features to retain from rfe selection
def rfe_fs(X, y, learner, step=2):
	rfe_selector = RFECV(learner, step=step);	
	X_reduced = rfe_selector.fit_transform(X, y);
	# print selected features - those that have a rank of 1
	print([feature for feature in rfe_selector.get_feature_names_out()]);
	return X_reduced;

#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import textblob
from textblob import TextBlob
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import nltk
nltk.download('wordnet')
nltk.download('wordnet_ic')
import numpy as np
import string
import pandas as pd

#df = pd.read_csv('qa.csv',index_col=0)
df = pd.read_csv('contexts.csv',index_col=0)

#docs = pd.DataFrame(np.unique(df['question']),columns=['question'])['question'][0:250]
docs = pd.DataFrame(np.unique(df['context']),columns=['context'])['context'][0:100]
tokens_ = [docs.iloc[c].split(' ') for c in np.array(range(0,len(docs),1))]
token_lengths = [len(t) for t in tokens_]

def createVocab(docList):
    vocab = {}
    for doc in docList:
        #print(doc)
        doc= doc.translate(str.maketrans('', '', string.punctuation))
        word_synsets = [wordnet.synsets(word) for word in word_tokenize(doc.lower())]
        
        for word_synset in word_synsets:
            if(word_synset):
                for lemma in word_synset[0].lemmas():
                    if(lemma.name() in vocab.keys()):
                        vocab[lemma.name()] = vocab[lemma.name()] +1
                    else:
                        vocab[lemma.name()] =1 
    return vocab

"""
docs=[ "Sachin is considered to be one of the greatest cricket players",
          "Federer is considered one of the greatest tennis players",
          "Nadal is considered one of the greatest tennis players",
          "Virat is the captain of the  Indian cricket team"
          
]
"""

vocab = createVocab(docs)

#Compute document term matrix as well idf for each term 

termDict={}

docsTFMat = np.zeros((len(docs),len(vocab)))

docsIdfMat = np.zeros((len(vocab),len(docs)))

docTermDf = pd.DataFrame(docsTFMat ,columns=sorted(vocab.keys()))
docCount=0
for doc in docs:
    doc= doc.translate(str.maketrans('', '', string.punctuation))
    word_synsets = [wordnet.synsets(word) for word in word_tokenize(doc.lower())]
    for word_synset in word_synsets:
        if (word_synset):
            for lemma in word_synset[0].lemmas():
                if(lemma.name() in vocab.keys()):
                    docTermDf[lemma.name()][docCount] = docTermDf[lemma.name()][docCount] +1
          
    docCount = docCount +1

feature_names = sorted(vocab.keys())
docList=np.array(range(0,len(docs),1))

#Computed idf for each word in vocab
idfDict={}

for column in docTermDf.columns:
    idfDict[column]= np.log((len(docs) +1 )/(1+ (docTermDf[column] != 0).sum()))+1
    
#compute tf.idf matrix
docsTfIdfMat = np.zeros((len(docs),len(vocab)))
docTfIdfDf = pd.DataFrame(docsTfIdfMat ,columns=sorted(vocab.keys()))

docCount = 0
for doc in docs:
    for key in idfDict.keys():
        docTfIdfDf[key][docCount] = docTermDf[key][docCount] * idfDict[key]
    docCount = docCount +1 

skDocsTfIdfdf = pd.DataFrame(docTfIdfDf, index=sorted(docList), columns=feature_names)
print(skDocsTfIdfdf)

from sklearn.metrics.pairwise import cosine_similarity

#compute cosine similarity
csim = cosine_similarity(docTfIdfDf, docTfIdfDf)

csimDf = pd.DataFrame(csim,index=sorted(docList),columns=sorted(docList))
csimDf

def derive_centroids(data, labels):
    # Calculate number of clusters
    n_clusters = len(np.unique(labels))
  
    # Create empty centroid array
    centroids = np.zeros((n_clusters, data.shape[1]))
    
    # Get indices of each cluster 
    for i in range(n_clusters):
        indices = np.where(labels == i)[0]
        
        # Calculate mean of each cluster and store it as centroid
        centroids[i] = np.nanmean(data.iloc[indices], axis = 0)
    
    return centroids

def bcss_tss_ratio(X, cluster_centers):

    # Computing Within Cluster Sum of Squares (WCSS)
    wcss = np.sum([np.nansum((x - cluster_centers[i]) ** 2) for i in range(len(cluster_centers)) for x in X[labels == i]])

    # Computing Between Cluster Sum of Squares (BCSS)
    bcss = np.sum([np.nansum((x - center) ** 2) for center in cluster_centers for x in X])

    # Computing Total Sum of Squares (TSS)
    tss = np.sum([np.nansum((x - np.nanmean(X)) ** 2) for x in X])

    # Computing the Ratio of WCSS to TSS
    wcss_tss_ratio = wcss/tss

    # Computing the Ratio of BCSS to TSS
    bcss_tss_ratio = bcss/tss

    #return wcss_tss_ratio, bcss_tss_ratio
    return bcss_tss_ratio

from sklearn.cluster import DBSCAN

label_sets = []

for e in np.array([1,2,3,5,8,13])/10:
    for m in np.array([1,2,3,5,8,13,21,34,55,89,144]):
        #print(e)
        model_DB = DBSCAN(eps = e, min_samples = m, metric = 'euclidean').fit(csimDf)
        labels = model_DB.labels_
        centroids = derive_centroids(csimDf,labels)
        label_sets.append([e,m,len(np.unique(labels)),labels,centroids,bcss_tss_ratio(csimDf,centroids)])
        
#sizes = [len(np.unique(l)) for l in label_sets]
#pd.DataFrame(sizes).hist()        

results = pd.DataFrame(label_sets)
results.columns = ['e','m','clusters','labels','centroids','ratio']

results.query('ratio > 0').sort_values('clusters',ascending=True)

chosen = 55
"""
questions = pd.DataFrame(docs)
questions['cluster'] = results.iloc[58]['labels']
"""
contexts = pd.DataFrame(docs)
contexts['cluster'] = results.iloc[chosen]['labels']

count = results.iloc[chosen]['clusters']

#set_ = questions.query('cluster >= 0').sort_values(by='cluster')
set_ = contexts.query('cluster >= 0').sort_values(by='cluster')
for c in np.array(range(0,count-1,1)):
    print('cluster: ', c)
    #print(set_[set_['cluster']==c]['question'].values)
    print(set_[set_['cluster']==c]['context'].values)

set_.to_csv('contexts.csv')
#set_.to_csv('questions.csv')

"""
#from nltk.corpus import wordnet as wn
#from nltk.corpus import wordnet_ic

dog=wn.synsets('dog', pos=wn.NOUN)[0] #get the first noun synonym of the word "dog"
print(dog)
cat=wn.synsets('cat', pos=wn.NOUN)[0]
rose=wn.synsets('rose', pos=wn.NOUN)[0]
flower=wn.synsets('flower', pos=wn.NOUN)[0]

brown_ic = wordnet_ic.ic('ic-brown.dat') #load the brown corpus to compute the IC

rose.res_similarity(flower, brown_ic)
rose.res_similarity(dog, brown_ic)
cat.res_similarity(dog, brown_ic)
"""

import pandas as pd

from top2vec import Top2Vec

#model = Top2Vec(documents)

contexts = pd.read_csv('contexts.csv',index_col=0)

subset

len(contexts)

def find_most_common_words(skDocsTfIdfdf, threshold):
    # Calculate the sums of the tf-idf values across all the documents
    tf_idf_sums = skDocsTfIdfdf.sum(axis=0)
    
    # Sort the tf-idf values in descending order
    sorted_tf_idf_sums = tf_idf_sums.sort_values(ascending=False)
    
    # Filter out the words with the highest tf-idf values, so that only the top 50% of values are displayed
    filtered_tf_idf_sums = sorted_tf_idf_sums[sorted_tf_idf_sums > sorted_tf_idf_sums.quantile(1-threshold)]
    return(filtered_tf_idf_sums)
    # Return the sorted dataframe

def createVocab_lemma(docs):
    vocab={}
    for doc in docs:
        doc= doc.translate(str.maketrans('', '', string.punctuation))
        for word in word_tokenize(doc.lower()):
            if word not in vocab.keys():
                vocab[word] = 1
                # Create a synonym list for each word
                word_synsets = [wordnet.synsets(word) for word in word_tokenize(doc.lower())]
                synonyms = []
                for word_synset in word_synsets:
                    if (word_synset):
                        for lemma in word_synset[0].lemmas():
                            synonyms.append(lemma.name())
                vocab[word] = synonyms
    return vocab

skDocsTfIdfdf

cluster_topics = []
import numpy as np
for c in np.array(range(0,len(np.unique(contexts['cluster'])),1)):
    print(c)
    docs = contexts[contexts['cluster']==c]['context'].values

    vocab = createVocab_lemma(docs)

    #Compute document term matrix as well idf for each term 

    termDict={}

    docsTFMat = np.zeros((len(docs),len(vocab)))

    docsIdfMat = np.zeros((len(vocab),len(docs)))

    docTermDf = pd.DataFrame(docsTFMat ,columns=sorted(vocab.keys()))
    docCount=0
    for doc in docs:
        doc= doc.translate(str.maketrans('', '', string.punctuation))
        word_synsets = [wordnet.synsets(word) for word in word_tokenize(doc.lower())]
        for word_synset in word_synsets:
            if (word_synset):
                for lemma in word_synset[0].lemmas():
                    if(lemma.name() in vocab.keys()):
                        docTermDf[lemma.name()][docCount] = docTermDf[lemma.name()][docCount] +1

        docCount = docCount +1

    feature_names = sorted(vocab.keys())
    docList=np.array(range(0,len(docs),1))

    #Computed idf for each word in vocab
    idfDict={}

    for column in docTermDf.columns:
        idfDict[column]= np.log((len(docs) +1 )/(1+ (docTermDf[column] != 0).sum()))+1

    #compute tf.idf matrix
    docsTfIdfMat = np.zeros((len(docs),len(vocab)))
    docTfIdfDf = pd.DataFrame(docsTfIdfMat ,columns=sorted(vocab.keys()))

    docCount = 0
    for doc in docs:
        for key in idfDict.keys():
            docTfIdfDf[key][docCount] = docTermDf[key][docCount] * idfDict[key]
        docCount = docCount +1 

    skDocsTfIdfdf = pd.DataFrame(docTfIdfDf, index=sorted(docList), columns=feature_names)
    cluster_topics.append(find_most_common_words(skDocsTfIdfdf,.5))
    """
    model = Top2Vec(subset.flatten(), embedding_model='universal-sentence-encoder')
    topic_sizes, topic_nums = model.get_topic_sizes()
    topic_words, word_scores, topic_nums = model.get_topics(model.get_num_topics())
    
    print(subset)
    print(topic_words)
    cluster_topics.append([subset,topic_sizes,topic_words, word_scores, topic_nums])
    """

# In[38]:

for c in np.array(range(0,len(np.unique(contexts['cluster'])),1)):
    print(c)
    print(cluster_topics[c])
    docs = contexts[contexts['cluster']==c]['context'].values
    print(docs)


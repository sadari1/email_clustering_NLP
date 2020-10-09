# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import pandas as pd 
import numpy as np
import time
from gensim import corpora
from collections import defaultdict
from gensim import models
from gensim import similarities
import secrets


# %%
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, StructField, StructType, StringType, IntegerType
from functional import seq
import time
import re
# old pyspark version is 2.4.5


# %%
def vec_to_sent(array):
    sent = ""
    for g in range(len(array)):
        sent = sent + array[g] + " "
    return sent

def create_corpus(documents):
    stoplist = set('for a of the and to in'.split())

    texts = [
        [word for word in document.lower().split() if word not in stoplist]
        for document in documents
    ]

    # remove words that appear only once
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    texts = [
        [token for token in text if frequency[token] > 1]
        for text in texts
    ]

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    return dictionary, corpus


# %%

# Reads the emails resulting from the first round of clustering. 
clustered_emails = pd.read_csv("milo/output/enron_clustered.csv")


# %%
unique_clusters = clustered_emails.Cluster.unique()


# %%

# This is the clustering algorithm for the second round.

tic = time.time()

# Sets all clusters to 0 initially.
clus_list_2 = [0 for k in range(len(clustered_emails))]

clustered_emails["chain_cluster"] = clus_list_2

for idx, clus in enumerate(unique_clusters[:20]):

    cluster_view = clustered_emails.loc[clustered_emails.Cluster == clus]

    documents = []

    # Parses the entity vector which got saved as a string when loaded by pandas.
    # TODO: save as a json instead so the arrays are preserved. 
    for f in range(len(cluster_view)):
        vector = cluster_view.iloc[f]["Entity Vector"].replace("[", "").replace("]", "").replace("'", "").split(",")
        sent = vec_to_sent(vector)
        documents.append(sent)

    dictionary, corpus = create_corpus(documents)

    # Builds an LSI model. This approach is faulty upon reivew, requires further investigation into the parameters.
    try:
        lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)
    except:
        dictionary, corpus = create_corpus(documents*2)
        lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)

    index = similarities.MatrixSimilarity(lsi[corpus]) 

    # Attempts to find matches based on the similarity of the entity vectors (Spacy's default NER-extracted entity list). 
    for row in range(len(cluster_view)):
        vec = cluster_view.iloc[row]["Entity Vector"].replace("[", "").replace("]", "").replace("'", "").split(",")
        doc = vec_to_sent(vec)

        vec_bow = dictionary.doc2bow(doc.lower().split())
        vec_lsi = lsi[vec_bow]
        
        sims = index[vec_lsi]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])

        email_match_list = []

        # Tweak this value with the LSI model parameters. 2 topics is too few.
        for i, s in enumerate(sims):
            if s[1] >= 0.99:
                email_match_list.append(s[0])
        
        # Generates a random ID to assign. 
        exists = True
        while exists:
            mini_cluster_id = secrets.randbits(32)
            if mini_cluster_id  not in unique_clusters or mini_cluster_id not in clus_list_2:
                exists = False

        # Assigns the mini-cluster
        for row_num in email_match_list:
            try:
                index_val = cluster_view.index[row_num]
                clustered_emails.loc[index_val, "chain_cluster"] = mini_cluster_id
                # cluster_view.iloc[row_num, -1] = mini_cluster_id
            except:
                continue
        
        print(f"Cluster # {idx + 1} -- row {row} / {len(cluster_view)}")

toc = time.time()

print(f"Job took {toc-tic} seconds")


# %%
# clustered_emails.to_csv("../output/enron_chain_clustered.csv", index=False)


# %%
# chain_clusters = clustered_emails.chain_cluster.unique()
# chain_clusters


# %%
# clustered_emails[clustered_emails.chain_cluster == 1455375456]


# %%
# clustered_emails[clustered_emails.Subject == "Managing Director and Vice President Elections"]


# %%
# test = pd.read_csv("../output/enron_chain_clustered.csv")


# %%
# test[test.chain_cluster == 2769081503]



# email_clustering_NLP

Using various NLP techniques on the Enron dataset to attempt to intelligently cluster the emails by chain as close to a human as possible.

# Enron Email Clusterer

This is the initial update as of 10-8-2020.

The run order of the code is described in "enron email matching.txt"
but is as follows:

1. enron_email_parsing_orig.py

2. enron_from_to_clustering.py

3. enron_clustered_matcher.py


The file described in 1. currently uses a rudimentary parser that misses a lot of edge cases. A better parser is currently in progress to account for several edge cases when extracting tokens. For now, the current parser is being utilized as it gets the correct data for a good number of emails.

The file described in 2. then attempts to pair up all emails based on
from-to using PySpark GraphFrames. These clusters are then assigned to all of the emails accordingly.

The file described in 3. attempts a second round of clustering, intending to assign cluster IDs to all of the emails filtered out by the initial from-to clustering. This would effectively create the desired clusters, where email chains have been linked together. Currently, this utilizes Gensim and Spacy NER vectors to attempt to discover the most similar documents.

This approach has not worked, so what is currently being worked on is each approach of the following (beginning with approach one), in an attempt to utilize deep learning to help us cluster these emails:

Approach one:

1. Run an autoencoder that simply takes in the input fields (from, to, cc, subject, feature vector of text) and attempts to encode and decode it, learning essentially a latent space representing all of the different encodings the autoencoder has seen.

2. Split the autoencoder into its encoder component and run the features of two emails in a siamese network containing these encoders, generating encodings of each email. Then, attempt to label a similarity score for these emails based on how likely they are to be alike.

Approach two:

1. Run a large model that takes two separate inputs of two different emails (from, to, cc, subject, feature vector of text), and runs the data through the network to result in a similarity score value that indicates whether the two emails match or not.


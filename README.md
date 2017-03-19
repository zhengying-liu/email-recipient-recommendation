# email-recipient-recommendation
This is the <a href="https://inclass.kaggle.com/c/master-data-science-mva-data-competition-2017">data challenge</a> for the course "Advanced Machine Learning for Graph and Text" in the Master Data Science program of Université Paris-Saclay.

In this challenge, we are asked to develop such a system, which, given the content and the date of a message, recommends a list of 10 recipients ranked by decreasing order of relevance.

Our team consists of four promising master students: Chia-Man, Mario, Salma and Zhengying.

We named our team "450euros", which is incidentally the amount of prize that will be awarded to the winning team.


### Run 

- run the script `code/install.sh`

- The input `.csv` data files should be in `code/input`

- The generated submission files will be in `code/output`


### project structure (in `code/`)

- `main.py`: contains the code allowing to reproduce all our submissions 

- `bow_knn.py`: implements a k-nearest-neighbors model with Bag-Of-Words features

- `word2vec_logistic_regression.py`: implements a multilabel Logistic Regression with word2vec, using only the recipients in the address book

- `tfidf_linearsvc.py`: implements a multilabel SVC with tf-idf features, using only the recipients in the address book

- `feature_extraction.py`: implements the Word2Vec feature extractor

- `evaluation.py` : contains the functions designed for evaluating the performance locally

- `*.ipynb` : notebooks containing the exploration of the emails, and analysis of the predictions outputted by our models


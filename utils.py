from nltk import word_tokenize
from tqdm import tqdm
import pandas as pd

import string
import re
import itertools
#import igraph
import nltk
import operator
from nltk.corpus import stopwords
# requires nltk 3.2.1
from nltk import pos_tag

def clean_text_simple(text, remove_stopwords=True, pos_filtering=True, stemming=True, keep_dash=True):
    if keep_dash:
        punct = string.punctuation.replace('-', '')
    else:
        punct = string.punctuation

    # convert to lower case
    text = text.lower()
    # remove punctuation (preserving intra-word dashes)
    text = ''.join(l for l in text if l not in punct)
    # strip extra white space
    text = re.sub(' +',' ',text)
    # strip leading and trailing white space
    text = text.strip()
    # tokenize (split based on whitespace)
    ### fill the gap ###
    tokens = text.split(' ')
    tokens = [token for token in tokens if token != '']
    if pos_filtering == True:
        tagged_tokens = pos_tag(tokens)
        tokens_keep = []
        for i in range(len(tagged_tokens)):
            item = tagged_tokens[i]
            if (
            item[1] == 'NN' or
            item[1] == 'NNS' or
            item[1] == 'NNP' or
            item[1] == 'NNPS' or
            item[1] == 'JJ' or
            item[1] == 'JJS' or
            item[1] == 'JJR'
            ):
                tokens_keep.append(item[0])
        tokens = tokens_keep
    if remove_stopwords:
        stpwds = stopwords.words('english')
        # remove stopwords
        tokens = [t for t in tokens if t not in stpwds]
    if stemming:
        stemmer = nltk.stem.PorterStemmer()
        # apply Porter's stemmer
        tokens_stemmed = list()
        for token in tokens:
            tokens_stemmed.append(stemmer.stem(token))
        tokens = tokens_stemmed

    return tokens

def read_data_info(filename="input/training_info.csv", nrows=20):
    info = pd.read_csv(filename, sep=',', header=0, nrows=nrows)
    info_dict = {}

    for index, series in tqdm(info.iterrows()):
        row = series.tolist()
        mid = row[0]
        date = row[1]
        body = row[2]
        recipients = None

        info_dict[mid] = {
            'date' : date,
            'body': clean_text_simple(body, stemming=False, keep_dash=False),
        }

        if len(row) > 3:
            recipients = row[3]
            recipients = recipients.split(' ')
            recipients = [rec for rec in recipients if '@' in rec]
            info_dict[mid]['recipients'] = recipients

    return info_dict

if __name__ == "__main__":
    print(read_data_info())

from nltk import word_tokenize
from tqdm import tqdm
import pandas as pd
import numpy as np
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
        
    text = clean_raw_text(text)
        
    # convert to lower case
    text = text.lower()
    # remove punctuation (preserving intra-word dashes)
    replace_punctuation = str.maketrans(punct, ' ' * len(punct))
    text = text.translate(replace_punctuation)
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

def read_data_info(filename="input/training_info.csv", nrows=None):
    if nrows is None:
        info = pd.read_csv(filename, sep=',', header=0)
    else:
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
    
def get_dataframes():
    """
    Construct four pd.DataFrame: training, training_info, test, test_info
    
    columns in training: ['sender', 'mids', 'list_of_mids', 'address_book']
    columns in training_info: ['mid', 'date', 'body', 'recipients', 
                               'list_of_recipients', 'sender']
    columns in test: ['sender', 'mids', 'list_of_mids', 'address_book']
    columns in test_info: ['mid', 'date', 'body', 'sender']
    
    """
    path_to_data = "input/"
    training = pd.read_csv(path_to_data + 'training_set.csv', sep=',')
    training_info = pd.read_csv(path_to_data + 'training_info.csv', sep=',')
    test = pd.read_csv(path_to_data + 'test_set.csv', sep=',')
    test_info = pd.read_csv(path_to_data + 'test_info.csv', sep=',')
    
    # list of mids in training, test
    print("Constructing list_of_mids...")
    func = lambda row: list(map(int, row.mids.split(' ')))
    training["list_of_mids"] = training.apply(func, axis=1)
    test["list_of_mids"] = test.apply(func, axis=1)
    
    # list of recipients in training_info
    print("Constructing list_of_recipients...")
    func = lambda row: [rec for rec in row.recipients.split(' ') if '@' in rec]
    training_info["list_of_recipients"] = training_info.apply(func, axis=1)
    
    # create an empty column for sender in training_info, test_info
    func = lambda row: ""
    training_info["sender"] = training_info.apply(func,axis=1)
    test_info["sender"] = test_info.apply(func,axis=1)

    # address book in training, test
    print("Constructing address book...")
    def count_contacted_recipients(row):
        list_of_mids = row["list_of_mids"]
        res = dict()
        for mid in list_of_mids:
            idx = np.where(training_info["mid"] == mid)[0][0]
            recipients = training_info.loc[idx]["list_of_recipients"]
            # add sender to training_info
            training_info.loc[idx, "sender"] = row.name
            for rec in recipients:
                if rec not in res:
                    res[rec] = 1
                else:
                    res[rec] += 1
        return res
    training["address_book"]= training.apply(count_contacted_recipients,axis=1)
    test["address_book"] = training["address_book"].copy()
    
    # add number of emails sent by each sender
    training["number_of_emails"] = training.apply(lambda row: len(row.list_of_mids), axis=1)
    
    # add number of recipients contacted by each sender
    training["number_of_recipients"] = training.apply(lambda row: len(row.address_book), axis=1)
    
    # add sender to test_info
    print("Add sender to test_info...")
    for index, row in test.iterrows():
        list_of_mids = row["list_of_mids"]
        for mid in list_of_mids:
            idx = np.where(test_info["mid"] == mid)[0][0]
            test_info.loc[idx, "sender"] = row.name
    
    return training, training_info, test, test_info

def received_mails_of_each_recipient_by_index(training_info_train):
    """
    Return a pd.DataFrame: mails_of_each_recipient
    index of mails_of_each_recipient: recipient
    columns in mails_of_each_recipient: ['list_of_messages_by_index',
                                         'number_of_received_messages']
    """
    print("Constructing received_mails_of_each_recipient_by_index...")
    recipient_mids_dict = dict()
    for index, row in training_info_train.iterrows():
        list_of_recipients = row["list_of_recipients"]
        for recipient in list_of_recipients:
            if recipient in recipient_mids_dict:
                recipient_mids_dict[recipient][0].append(index)
            else:
                recipient_mids_dict[recipient] = [[index]]
    mails_of_each_recipient = pd.DataFrame(recipient_mids_dict).T
    mails_of_each_recipient.columns = ['list_of_messages_by_index']
    mails_of_each_recipient['number_of_received_messages'] =\
                           mails_of_each_recipient.apply(lambda row: 
                               len(row['list_of_messages_by_index']),axis=1)                        
    return mails_of_each_recipient

def clean_raw_text(raw_text):
    """
    Given a string raw_text (e.g. the body of a mail), clean it and return a 
    string.
    """
    
    # First remove inline JavaScript/CSS:
    text = re.sub(r"(?is)<(script|style).*?>.*?()", " ", raw_text)
    # Remove url
    text = re.sub(r"^https?:\/\/.*[\r\n]*", " ", text, flags=re.MULTILINE)
    # Then remove html comments. 
    text = re.sub(r"(?s)[\n]?", "", text)
    # Next remove the remaining tags:
    text = re.sub(r"(?s)<.*?>", " ", text)
    # Remove forward/senders/time information, keep subject
    text = re.sub(r"-{2,}.*?Subject:", " ", text)
    text = re.sub(r"From:.*?Subject:", " ", text)
    text = re.sub(r"To:.*?Subject:", " ", text)
    text = re.sub(r"TO:.*?Subject:", " ", text)
    # Add a space before some uppercase letter, e.g. "ThanksLisa" => "Thanks Lisa"
    text = re.sub(r"(?=[A-Z])(?<=[^A-Z])(?<! )(?<!-)", " ", text)
    # Remove consecutive non letter string: "713/853-4218", "$80,400.00"
    text = re.sub(r"[^a-zA-Z]{5,}", " ", text)
    # Replace "\t" by " "
    text = re.sub(r"\t", " ", text)
    # replace "/"
    text = re.sub(r"/", " ", text)
    # replace " -" by " "
    text = re.sub(r" -+", " ", text)
    # Dealing with multiple white space
    text = re.sub(r" +", " ", text)
    # strip leading and trailing white space
    text = text.strip()
    return text

def find_common_recipients(training):
    list_of_address_book = []
    for index, row in training.iterrows():
        list_of_address_book.append(row.address_book)
    recipients_in_several_address_book = {}
    n = len(list_of_address_book)
    for i in range(n):
        for key in list_of_address_book[i]:
            if key not in recipients_in_several_address_book:
                for j in range(i+1,n):
                    if key in list_of_address_book[j]:
                        if key not in recipients_in_several_address_book:
                            recipients_in_several_address_book[key] = {i,j}
                        else:
                            recipients_in_several_address_book[key].add(j)
    return list_of_address_book, recipients_in_several_address_book

if __name__ == "__main__":
   list_of_address_book, recipients_in_several_address_book =\
       find_common_recipients(training)

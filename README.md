# email-recipient-recommendation
This is the <a href="https://inclass.kaggle.com/c/master-data-science-mva-data-competition-2017">data challenge</a> for the course "Advanced Machine Learning for Graph and Text" in the Master Data Science program of Université Paris-Saclay.

In this challenge, we are asked to develop such a system, which, given the content and the date of a message, recommends a list of 10 recipients ranked by decreasing order of relevance.

Our team consists of four promising master students: Chia-Man, Mario, Salma and Zhengying.

We name our team "450euros", which is accidentally the amount of prize that will be awarded to the winning team.


## TODO: 

- Features

    1. Normalized sent frequency: the number of
    messages sent to this recipient divided by the total number of messages sent by this particular user 
    
    2. Normalized received frequency : the number of messages
    received from this recipient divided by the total number of messages received by this particular user
    
    3. co-occurrence of recipients on other messages in the training set: Given a message
    with three recipients a1, a2 and a3, let the frequency of co-occurrence between recipients
    a1 and a2 be F(a1, a2) (i.e., the number of messages in the training set that had a1 as
    well as a2 as recipients). Then the relative co-occurrence frequency of users a1, a2 and
    a3 will be proportional to, respectively, F(a1, a2) + F(a1, a3), F(a2, a3) + F(a2, a1)
    and F(a3, a1) + F(a3, a2): i.e., the relative co-occurrence frequency of each recipient
    ai = sum_{i \neq j} F(ai, aj ). These values are then divided by their sum and normalized to one.
    In case of two recipients only, the value of this feature is obviously 0.5 for each.
    
    4. Recency features: we extract the normalized sent frequency of all users in the training set. But instead of
        using the entire training set for the extraction, we only use the last 20, last 50 and last 100 messages.
    
Reference: http://www.cs.cmu.edu/~vitor/thesis/carvalho_thesis.pdf, pages 49-50-51
    
- Model : 

Reference: http://www.cs.cmu.edu/~wcohen/postscript/cc-predict-submitted.pdf

    1. Centroid (section 3.1.1)
    
    2. K-NN (section 3.1.2)
    
    3. Cross-Validation (section 3.2.1)
    
    4. Classifier (section 3.2.3)
    
Reference : 
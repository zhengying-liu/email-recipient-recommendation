from baseline_zhengying import predict_by_nearest_recipients, received_mails_of_each_recipient_by_index, get_bag_words
from utils import get_dataframes
import multilabel
import multilabel_tfidf

training, training_info, test, test_info = get_dataframes()

# Solution with word2vec, Logistic Regression and address book
pred_test, models_test = multilabel.predict_by_multilabel_for_each_sender(training_info, test_info, training)
pred_test = pred_test[['mid', 'recipients']]
pred_test.to_csv("pred_logistic_regression.txt", index=False)

# Solution with with tf-idf, LinearSVC and address book
pred_test, models_test = multilabel_tfidf.predict_by_multilabel_for_each_sender(training_info, test_info)
pred_test['recipients'] = pred_test.apply(lambda row: " ".join(row["list_of_recipients"]), axis=1)
pred_test = pred_test[['mid', 'recipients']]
pred_test.to_csv("pred_tf-idf_LinearSVC.csv", index=False)

# Solution with prediction by nearest recipient
mails_of_each_recipient = received_mails_of_each_recipient_by_index(training_info)
X_train, X_test, count_vect = get_bag_words(training_info, training_info)
build_char_vector(X_train, mails_of_each_recipient)
pred = predict_by_nearest_recipients(mails_of_each_recipient, training_info, count_vect, training)
pred.to_csv("pred_nearest_recipient.csv", index=False)

from utils import get_dataframes
import multilabel
import multilabel_tfidf

training, training_info, test, test_info = get_dataframes()

# Solution with word2vec, Logistic Regression and address book
#pred_test, models_test = multilabel.predict_by_multilabel_for_each_sender(training_info, test_info, training)
#pred_test = pred_test[['mid', 'recipients']]
#pred_test.to_csv("pred_logistic_regression.txt", index=False)

# Solution with with tf-idf, LinearSVC and address book
pred_test, models_test = multilabel_tfidf.predict_by_multilabel_for_each_sender(training_info, test_info)
pred_test['recipients'] = pred_test.apply(lambda row: " ".join(row["list_of_recipients"]), axis=1)
pred_test = pred_test[['mid', 'recipients']]
pred_test.to_csv("pred_tf-idf_LinearSVC.csv", index=False)

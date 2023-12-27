import pickle
import os 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from keras.layers import Dense, Dropout, Lambda, Embedding, Conv1D, LSTM, Input
from tensorflow import keras
from sklearn.metrics import accuracy_score

from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
import sklearn
from sklearn.model_selection import train_test_split



#Test với input user nhập tay
text_test ='mới dùng được 2 lần là đã hỏng'
text_label = 'positive'




X_data = pickle.load(open('X_data.pkl', 'rb'))
y_data = pickle.load(open('y_data.pkl', 'rb'))
X_test = pickle.load(open('X_data_test.pkl', 'rb'))
X_test.append(text_test)
y_test = pickle.load(open('y_data_test.pkl', 'rb'))
y_test.append(text_label)


# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(X_data)

# transform the training and validation data using count vectorizer object
X_data_count = count_vect.transform(X_data)
X_test_count = count_vect.transform(X_test)

# word level - we choose max number of words equal to 30000 except all words (100k+ words)
tfidf_vect = TfidfVectorizer(analyzer='word', max_features=30000)
tfidf_vect.fit(X_data) # learn vocabulary and idf from training set
X_data_tfidf =  tfidf_vect.transform(X_data)
# assume that we don't have test set before
X_test_tfidf =  tfidf_vect.transform(X_test)

# # ngram level - we choose max number of words equal to 30000 except all words (100k+ words)
# tfidf_vect_ngram = TfidfVectorizer(analyzer='word', max_features=30000, ngram_range=(2, 3))
# tfidf_vect_ngram.fit(X_data)
# X_data_tfidf_ngram =  tfidf_vect_ngram.transform(X_data)
# # assume that we don't have test set before
# X_test_tfidf_ngram =  tfidf_vect_ngram.transform(X_test)

# # ngram-char level - we choose max number of words equal to 30000 except all words (100k+ words)
# tfidf_vect_ngram_char = TfidfVectorizer(analyzer='char', max_features=30000, ngram_range=(2, 3))
# tfidf_vect_ngram_char.fit(X_data)
# X_data_tfidf_ngram_char =  tfidf_vect_ngram_char.transform(X_data)
# # assume that we don't have test set before
# X_test_tfidf_ngram_char =  tfidf_vect_ngram_char.transform(X_test)

svd = TruncatedSVD(n_components=300, random_state=42)
svd.fit(X_data_tfidf)


X_data_tfidf_svd = svd.transform(X_data_tfidf)
X_test_tfidf_svd = svd.transform(X_test_tfidf)

# from gensim.models import KeyedVectors 
# dir_path = os.path.dirname(os.path.realpath(os.getcwd()))
# word2vec_model_path = os.path.join(dir_path, "Data/vi/vi.vec")

# w2v = KeyedVectors.load_word2vec_format(word2vec_model_path)
# vocab = w2v.wv.vocab
# wv = w2v.wv

# def get_word2vec_data(X):
#     word2vec_data = []
#     for x in X:
#         sentence = []
#         for word in x.split(" "):
#             if word in vocab:
#                 sentence.append(wv[word])

#         word2vec_data.append(sentence)

#     return word2vec_data

# X_data_w2v = get_word2vec_data(X_data)
# X_test_w2v = get_word2vec_data(X_test)

encoder = preprocessing.LabelEncoder()
y_data_n = encoder.fit_transform(y_data)
y_test_n = encoder.fit_transform(y_test)
y_data = encoder.fit_transform(y_data)
y_test = encoder.fit_transform(y_test)

encoder.classes_

def train_model(classifier, X_data, y_data, X_test, y_test, is_neuralnet=False, n_epochs=3):       
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.1, random_state=42)
    
    if is_neuralnet:
        classifier.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=n_epochs, batch_size=256)
        
        val_predictions = classifier.predict(X_val)
        test_predictions = classifier.predict(X_test)
        val_predictions = val_predictions.argmax(axis=-1)
        test_predictions = test_predictions.argmax(axis=-1)
        print(test_predictions)
        filename = 'cnn.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(classifier, file)
    else:
        classifier.fit(X_train, y_train)
        train_predictions = classifier.predict(X_train)
        val_predictions = classifier.predict(X_val)
        test_predictions = classifier.predict(X_test)


        
        # print('Kết quả với mô hình naive bayes input: ' + text_test+ ' là: Nhãn '+str(test_predictions[len(test_predictions)-1]))
        # print(test_predictions)
        # r = accuracy_score(y_train, train_predictions)
        filename = 'naive_bayes_model.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(classifier, file)

# get the accuracy score of the test data. 

        
    print("Validation accuracy: ", sklearn.metrics.accuracy_score(val_predictions, y_val))
    print("Test accuracy: ", sklearn.metrics.accuracy_score(test_predictions, y_test))

# train_model(BernoulliNB(), X_data_tfidf, y_data, X_test_tfidf, y_test, is_neuralnet=False)


# mạng nơ ron nhân tạo
def create_dnn_model():
    input_layer = Input(shape=(300,))
    layer = Dense(1024, activation='relu')(input_layer)
    layer = Dense(1024, activation='relu')(layer)
    layer = Dense(512, activation='relu')(layer)
    output_layer = Dense(10, activation='softmax')(layer)
    
    classifier = keras.Model(input_layer, output_layer)
    classifier.compile(optimizer=keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    train_model(classifier=classifier, X_data=X_data_tfidf_svd, y_data=y_data_n, X_test=X_test_tfidf_svd, y_test=y_test_n, is_neuralnet=True)
#chạy naive bayes
train_model(MultinomialNB(), X_data_tfidf, y_data, X_test_tfidf, y_test, is_neuralnet=False)
#chạy cnn
# create_dnn_model()

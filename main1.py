import pickle
import os 
from gensim.models import Word2Vec
from keras.layers import Dense, Input
from tensorflow import keras
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import preprocessing
import numpy as np
import sklearn
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

text_test ='mới dùng được 2 lần là đã hỏng'
text_label = 'positive'

X_data = pickle.load(open('X_data.pkl', 'rb'))
y_data = pickle.load(open('y_data.pkl', 'rb'))
X_test = pickle.load(open('X_data_test.pkl', 'rb'))
X_test.append(text_test)
y_test = pickle.load(open('y_data_test.pkl', 'rb'))
y_test.append(text_label)

# Đọc mô hình Word2Vec từ file đã được đào tạo hoặc tải về từ nguồn nào đó
# word2vec_model = Word2Vec.load('./MiAI_Word2Vec_Demo\\data\\word_model.save')
word2vec_model = Word2Vec.load('./word2model.save')

# Chuyển đổi văn bản thành vectơ Word2Vec
def text_to_w2v_vector(text, model):
    words = text.split()
    vectorized_words = [model.wv[word] for word in words if word in model.wv]
    if vectorized_words:
        return np.mean(vectorized_words, axis=0)
    else:
        return np.zeros(model.vector_size)  # Vectơ không nếu không có từ nào trong từ điển

X_data_w2v = np.array([text_to_w2v_vector(text, word2vec_model) for text in X_data])
X_test_w2v = np.array([text_to_w2v_vector(text, word2vec_model) for text in X_test])
encoder = preprocessing.LabelEncoder()
y_data_n = encoder.fit_transform(y_data)
y_test_n = encoder.fit_transform(y_test)
def train_model(classifier, X_data, y_data, X_test, y_test, is_neuralnet=False, n_epochs=5):       
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.1, random_state=42)
    
    if is_neuralnet:
        classifier.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=n_epochs, batch_size=256)
        
        val_predictions = classifier.predict(X_val)
        test_predictions = classifier.predict(X_test)
        val_predictions = val_predictions.argmax(axis=-1)
        test_predictions = test_predictions.argmax(axis=-1)
        filename = 'cnn.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(classifier, file)
        print("-- CNN --")
    else:
        classifier.fit(X_train, y_train)
        train_predictions = classifier.predict(X_train)
        val_predictions = classifier.predict(X_val)
        test_predictions = classifier.predict(X_test)
        print('Kết quả với mô hình MultinomialNB input: ' + text_test+ ' là: Nhãn '+test_predictions[len(test_predictions)-1])
        # r = accuracy_score(y_train, train_predictions)
        filename = 'MultinomialNB.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(classifier, file)
        print("-- NB --")

# get the accuracy score of the test data. 

        
    print("Validation accuracy: ", sklearn.metrics.accuracy_score(val_predictions, y_val))
    print("Test accuracy: ", sklearn.metrics.accuracy_score(test_predictions, y_test))
# Sử dụng mô hình phân loại hiện tại (MultinomialNB) với vectơ Word2Vec
def create_dnn_model():
    # X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.1, random_state=42)
    input_layer = Input(shape=(100,))
    # input_layer = Input(shape=(300,))
    layer = Dense(1024, activation='relu')(input_layer)
    layer = Dense(1024, activation='relu')(layer)
    layer = Dense(512, activation='relu')(layer)
    output_layer = Dense(3, activation='softmax')(layer)
    
    classifier = keras.Model(input_layer, output_layer)
    classifier.compile(optimizer=keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    train_model(classifier=classifier, X_data=X_data_w2v, y_data = y_data_n, X_test=X_test_w2v, y_test=y_test_n, is_neuralnet=True)

train_model(BernoulliNB(), X_data_w2v, y_data, X_test_w2v, y_test, is_neuralnet=False)
create_dnn_model()

# Sử dụng mô hình phân loại SVM với vectơ Word2Vec
def create_svm_model():
    # Tạo và huấn luyện mô hình SVM
    svm_classifier = SVC(kernel='linear')  # Bạn có thể thay đổi kernel tùy thuộc vào yêu cầu
    svm_classifier.fit(X_train, y_train)  # Sử dụng tập huấn luyện đã được chia

    # Đánh giá mô hình trên tập validation và test
    val_predictions = svm_classifier.predict(X_val)
    test_predictions = svm_classifier.predict(X_test_w2v)

    print("-- SVM --")
    print("Validation accuracy: ", sklearn.metrics.accuracy_score(val_predictions, y_val))
    print("Test accuracy: ", sklearn.metrics.accuracy_score(test_predictions, y_test_n))

    # Lưu mô hình
    filename = 'svm_model.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(svm_classifier, file)

# Chia tập dữ liệu thành tập huấn luyện và tập validation
X_train, X_val, y_train, y_val = train_test_split(X_data_w2v, y_data_n, test_size=0.1, random_state=42)

# Gọi hàm tạo và huấn luyện mô hình SVM
create_svm_model()

# Tạo và huấn luyện mô hình LSTM
vocab_size = len(word2vec_model.wv)
embedding_dim = 100 
input_shape = X_train.shape[1]

lstm_model = Sequential()
lstm_model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=X_train.shape[1]))
lstm_model.add(LSTM(100))  
lstm_model.add(Dense(3, activation='softmax'))
lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=256)

# Đánh giá mô hình trên tập validation và test
val_predictions = lstm_model.predict(X_val)
test_predictions = lstm_model.predict(X_test_w2v)

val_predictions = val_predictions.argmax(axis=-1)
test_predictions = test_predictions.argmax(axis=-1)

print("-- LSTM --")
print("Validation accuracy: ", sklearn.metrics.accuracy_score(val_predictions, y_val))
print("Test accuracy: ", sklearn.metrics.accuracy_score(test_predictions, y_test_n))

# Lưu mô hình
lstm_model.save('lstm_model.h5')

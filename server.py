from flask import Flask, jsonify, request

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
from flask_cors import CORS


# text_test ='mới dùng được 2 lần là đã hỏng'
text_label = 'positive'
labelText = ["Tiêu cực", "Trung tính",  "Tích cực"]



X_data = pickle.load(open('X_data.pkl', 'rb'))
y_data_ = pickle.load(open('y_data.pkl', 'rb'))
X_test = pickle.load(open('X_data_test.pkl', 'rb'))
y_test_ = pickle.load(open('y_data_test.pkl', 'rb'))
y_test_.append(text_label)

def train_model(classifier, X_data, y_data, X_test, y_test, is_neuralnet=False, n_epochs=3):       
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.1, random_state=42)
    classifier.fit(X_train, y_train)
    train_predictions = classifier.predict(X_train)
    val_predictions = classifier.predict(X_val)
    test_predictions = classifier.predict(X_test)
    # return 'Kết quả với mô hình là: Nhãn '+str(test_predictions[len(test_predictions)-1])
    return 'Kết quả với mô hình là: Nhãn '+str(labelText[test_predictions[len(test_predictions)-1]])



app = Flask(__name__)
CORS(app)

@app.route('/', methods=['POST'])
def get_data():


    try:
        # Trích xuất dữ liệu từ body của yêu cầu
        data = request.get_json()

        # Kiểm tra xem 'input' có trong dữ liệu không
        if 'input' in data:
            input_data = data['input']
            X_test.append(input_data)
            count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
            count_vect.fit(X_data)


            tfidf_vect = TfidfVectorizer(analyzer='word', max_features=30000)
            tfidf_vect.fit(X_data) # learn vocabulary and idf from training set
            X_data_tfidf =  tfidf_vect.transform(X_data)
            X_test_tfidf =  tfidf_vect.transform(X_test)


            svd = TruncatedSVD(n_components=300, random_state=42)
            svd.fit(X_data_tfidf)



            encoder = preprocessing.LabelEncoder()
            y_data = encoder.fit_transform(y_data_)
            y_test = encoder.fit_transform(y_test_)

            encoder.classes_
            a = train_model(MultinomialNB(), X_data_tfidf, y_data, X_test_tfidf, y_test, is_neuralnet=False)
            return jsonify({'result': a})
        else:
            # Trả về lỗi nếu 'input' không được tìm thấy trong dữ liệu
            return jsonify({'error': 'Input not found in request data'}), 400

    except Exception as e:
        # Trả về lỗi nếu có lỗi xảy ra trong quá trình xử lý
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=False)

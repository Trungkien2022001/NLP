from flask import Flask, jsonify, request
import numpy as np
import pickle
import os 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from keras.layers import Dense, Dropout, Lambda, Embedding, Conv1D, LSTM, Input
from tensorflow import keras
from sklearn.metrics import accuracy_score
from gensim.models import Word2Vec

from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
import sklearn
from sklearn.model_selection import train_test_split
from flask_cors import CORS


# text_test ='mới dùng được 2 lần là đã hỏng'
labelText = ["Tiêu cực", "Trung tính",  "Tích cực"]

current_script_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_script_path)
resources_folder_path = os.path.join(current_directory)

X_data = pickle.load(open('X_data_train.pkl', 'rb'))
y_data_ = pickle.load(open('y_data_train.pkl', 'rb'))
X_test = pickle.load(open('X_data_test.pkl', 'rb'))
y_test_ = pickle.load(open('y_data_test.pkl', 'rb'))

word2vec_model = Word2Vec.load( os.path.join(resources_folder_path,'word2model.save'))


def text_to_w2v_vector(text, model):
            words = text.split()
            vectorized_words = [model.wv[word] for word in words if word in model.wv]
            if vectorized_words:
                return np.mean(vectorized_words, axis=0)
            else:
                return np.zeros(model.vector_size)  # Vectơ không nếu
X_data_ = np.array([text_to_w2v_vector(text, word2vec_model) for text in X_data])
encoder = preprocessing.LabelEncoder()
y_data = encoder.fit_transform(y_data_)




count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(X_data)


vect = TfidfVectorizer(analyzer='word', max_features=30000)
vect.fit(X_data) # learn vocabulary and idf from training set
X_data_ =  vect.transform(X_data)
svd = TruncatedSVD(n_components=300, random_state=42)
svd.fit(X_data_)

def train_model(classifier, X_data, y_data, X_test, y_test, input_data , is_neuralnet=False, n_epochs=3):       
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.1, random_state=42)
    if is_neuralnet:
            classifier.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=n_epochs, batch_size=256)
            
            val_predictions = classifier.predict(X_val)
            test_predictions = classifier.predict(X_test)
            val_predictions = val_predictions.argmax(axis=-1)
            test_predictions = test_predictions.argmax(axis=-1)
            print(test_predictions)
            filename = 'ann.pkl'
            with open(filename, 'wb') as file:
                pickle.dump(classifier, file)
    else:
        # alphas = [0.1, 0.5, 1.0, 2.0, 5.0]
        # fit_priors = [True, False]
        # class_priors = [None, [0.3, 0.6, 0.1], [0.2, 0.7, 0.1], [0.4, 0.4, 0.2]]

        # # Sử dụng GridSearchCV để tìm giá trị alpha tốt nhất
        # param_grid = {'alpha': alphas, 'fit_prior': fit_priors, 'class_prior': class_priors}
        # grid_search = GridSearchCV(classifier, param_grid, cv=5)
        # grid_search.fit(X_train, y_train)

        # # In ra giá trị alpha tốt nhất
        # best_alpha = grid_search.best_params_['alpha']
        # print("Best alpha:", best_alpha)
        # best_fit_prior = grid_search.best_params_['fit_prior']
        # print("Best fit_prior:", best_fit_prior)
        # best_class_prior = grid_search.best_params_['class_prior']
        # print("Best class_prior:", best_class_prior)
        classifier.fit(X_train, y_train)
        train_predictions = classifier.predict(X_train)
        val_predictions = classifier.predict(X_val)
        test_predictions = classifier.predict(X_test)
    # return 'Kết quả với mô hình là: Nhãn '+str(test_predictions[len(test_predictions)-1])
   
    result_array = []

    for i in range(len(input_data)):
        result_dict = {
            'key': input_data[i],
            'label': str(labelText[test_predictions[len(test_predictions) - len(input_data) + i]])
        }
        result_array.append(result_dict)

    return result_array



app = Flask(__name__)
CORS(app)

@app.route('/', methods=['POST'])
def get_data():


    try:
        data = request.get_json()

        if 'input' in data:
            input_data = data['input']
            for i in input_data:
                X_test.append(i)
            X_test_t =  vect.transform(X_test)

            def text_to_w2v_vector(text, model):
                words = text.split()
                vectorized_words = [model.wv[word] for word in words if word in model.wv]
                if vectorized_words:
                    return np.mean(vectorized_words, axis=0)
                else:
                    return np.zeros(model.vector_size)  # Vectơ không nếu không có từ nào trong từ điển
            # X_data_ = np.array([text_to_w2v_vector(text, word2vec_model) for text in X_data])
            # X_test_ = np.array([text_to_w2v_vector(text, word2vec_model) for text in X_test])

            y_test = encoder.fit_transform(y_test_)

            encoder.classes_
            a = train_model(MultinomialNB(), X_data_, y_data, X_test_t, y_test, input_data, is_neuralnet=False)
            return jsonify({'result': a})
        else:
            return jsonify({'error': 'Input not found in request data'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/model', methods=['POST'])
def get_data1():


    try:
        data = request.get_json()

        if 'input' in data:
            input_data = data['input']
            print(input_data)
            t=word2vec_model.wv.most_similar(input_data)
            return jsonify({'result': t})
           
        else:
            return jsonify({'error': 'Input not found in request data'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 200

if __name__ == '__main__':
    app.run(port=5000, debug=False)

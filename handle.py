from pyvi import ViTokenizer, ViPosTagger # thư viện NLP tiếng Việt
from tqdm import tqdm
import numpy as np
import gensim # thư viện NLP
import os 
import pickle
dir_path = os.path.dirname(os.path.realpath(os.getcwd()))
dir_path = os.path.join(dir_path)
stop_words = ['ma', 'anh', 'em', 'vì', 'thế', 'nhưng']
def get_data(folder_path):
    X = []
    y = []
    dirs = os.listdir(folder_path)
    for path in tqdm(dirs):
        file_paths = os.listdir(os.path.join(folder_path, path))
        for file_path in tqdm(file_paths):
            with open(os.path.join(folder_path, path, file_path), 'r', encoding="utf-8") as f:
                lines = f.readlines()
                lines = ' '.join(lines)
                lines = gensim.utils.simple_preprocess(lines)
                lines =  [word for word in lines if word.casefold() not in stop_words]
                lines = ' '.join(lines)
                lines = ViTokenizer.tokenize(lines)

                X.append(lines)
                y.append(path)

    return X, y
# train_path = os.path.join('E:\\Code\\Project\\NLP\\train_data') #train

# chỗ này thay bằng folder của project trên máy nhé (cộng thêm \\test_data nữa)
train_path = os.path.join('E:\\Code\\Project\\NLP\\test_data') #train
X_data, y_data = get_data(train_path)
pickle.dump(X_data, open('X_data_test.pkl', 'wb'))
pickle.dump(y_data, open('y_data_test.pkl', 'wb'))
print(X_data, y_data)
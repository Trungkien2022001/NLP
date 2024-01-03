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
    # Lấy danh sách thư mục con
    dirs = os.listdir(folder_path)
    # Duyệt qua từng thư mục con
    for path in tqdm(dirs):
        # Lấy danh sách các tệp trong thư mục con
        file_paths = os.listdir(os.path.join(folder_path, path))
        for file_path in tqdm(file_paths):
            with open(os.path.join(folder_path, path, file_path), 'r', encoding="utf-8") as f:
                lines = f.readlines()
                # chuyển danh sách các dòng thành một chuỗi
                lines = ' '.join(lines)
                # Chuyển đổi văn bản thành chữ thường
                # Tách văn bản thành các từ
                # Loại bỏ ký tự không cần thiết: dấu câu, ký tự đặc biệt, ký tự không mong muốn
                # Lọc từ dừng (stop words)
                # Giữ lại từ có ý nghĩa 
                lines = gensim.utils.simple_preprocess(lines)
                lines =  [word for word in lines if word.casefold() not in stop_words]
                lines = ' '.join(lines)
                lines = ViTokenizer.tokenize(lines)

                X.append(lines)
                y.append(path)

    return X, y
# train_path = os.path.join('./train_data') #train

# chỗ này thay bằng folder của project trên máy nhé (cộng thêm \\test_data nữa)
train_path = os.path.join('./test_data') #train
X_data, y_data = get_data(train_path)
# Lưu
pickle.dump(X_data, open('X_data_test.pkl', 'wb'))
pickle.dump(y_data, open('y_data_test.pkl', 'wb'))
print(X_data, y_data)
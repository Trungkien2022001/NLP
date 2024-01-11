from pyvi import ViTokenizer, ViPosTagger # thư viện NLP tiếng Việt
from tqdm import tqdm
import numpy as np
import gensim # thư viện NLP
import os 
import pickle
dir_path = os.path.dirname(os.path.realpath(os.getcwd()))
dir_path = os.path.join(dir_path)
stop_words = [
    "và", "hay", "là", "của", "được", "rằng", "cho", "trong", "đến", "có",
    "đã", "cùng", "với", "như", "này", "một", "nhiều", "để", "từ", "khi",
    "đó", "làm", "ở", "được", "qua", "nếu", "nhưng", "về", "ở", "rất", "cũng",
    "được", "vậy", "bạn", "đều", "không", "đều", "vào", "ra", "nếu", "ai",
    "làm", "chỉ", "muốn", "cũng", "ở", "từ", "trên", "nếu", "là", "có", "không",
    "cho", "này", "cảm", "ơn", "chúng", "ta", "điều", "này", "điều", "nọ",
    "tại", "đây", "còn", "người", "ta", "có", "mình", "mình", "này", "một số",
    "những", "có", "rằng", "nhất", "họ", "nó", "có", "đó", "vào", "lúc",
    "sau", "ra", "làm", "việc", "càng", "mà", "được", "chứ", "ngay", "nhà",
    "đầu", "làm", "cái", "mới", "sự", "ngày", "thường", "lên", "đi", "nước",
    "nhiều", "làm", "việc", "mỗi", "làm", "cả", "nói", "người", "những",
    "có", "hơn", "của", "trước", "những", "cái", "người", "thế", "giới"
]

reverse_words = ["không",  "chẳng", "chưa"]
# s_words=["xấu", "đẹp", "tệ", "tuyệt_vời"]
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
                lines = [word for word in lines if word.casefold() not in stop_words]
                lines = gensim.utils.simple_preprocess(lines)

                new_lines = []
                i = 0
                while i < len(lines):
                    current_word = lines[i]

                    if current_word in reverse_words:
                        # Nếu là từ phủ định, kết hợp với từ tiếp theo để tạo từ ghép mới
                        if i + 1 < len(lines):
                            combined_word = current_word + '_' + lines[i + 1]
                            new_lines.append(combined_word)
                            i += 2  # Bước 2 để bỏ qua từ tiếp theo đã được kết hợp
                        else:
                            new_lines.append(current_word)
                            i += 1
                    else:
                        new_lines.append(current_word)
                        i += 1

                lines = ' '.join(new_lines)
                lines = ViTokenizer.tokenize(lines)

                X.append(lines)
                y.append(path)

    return X, y
# train_path = os.path.join('./train_data') #train

# chỗ này thay bằng folder của project trên máy nhé (cộng thêm \\test_data nữa)
train_path = os.path.join('./train_data') #train
X_data, y_data = get_data(train_path)
pickle.dump(X_data, open('X_data_train.pkl', 'wb'))
pickle.dump(y_data, open('y_data_train.pkl', 'wb'))
print(X_data, y_data)
from gensim.models import Word2Vec
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten
# from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Bước 1: Chuẩn bị dữ liệu
positive_text1 = open('E:\\Code\\Project\\NLP\\data_raw\\test_nhan_0.txt', 'r', encoding='utf-8').read()
negative_text1 = open('E:\\Code\\Project\\NLP\\data_raw\\test_nhan_1.txt', 'r', encoding='utf-8').read()
neutral_text1 = open('E:\\Code\\Project\\NLP\\data_raw\\test_nhan_2.txt', 'r', encoding='utf-8').read()
positive_text = open('E:\\Code\\Project\\NLP\\data_raw\\train_nhan_0.txt', 'r', encoding='utf-8').read()
negative_text = open('E:\\Code\\Project\\NLP\\data_raw\\train_nhan_1.txt', 'r', encoding='utf-8').read()
neutral_text = open('E:\\Code\\Project\\NLP\\data_raw\\train_nhan_2.txt', 'r', encoding='utf-8').read()

def preprocess_text(text):
    # Chuyển đổi về chữ thường
    text = text.lower()
    
    # Loại bỏ kí tự đặc biệt
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])

    # Loại bỏ dấu

    words = [word for word in text.split(' ') if word]

    return words

# Bước 2: Đào tạo mô hình Word2Vec
tokenized_positive = [preprocess_text(sentence) for sentence in positive_text.splitlines()]
tokenized_negative = [preprocess_text(sentence) for sentence in negative_text.splitlines()]
tokenized_neutral = [preprocess_text(sentence) for sentence in neutral_text.splitlines()]
# Bước 2: Đào tạo mô hình Word2Vec
tokenized_positive1 = [preprocess_text(sentence) for sentence in positive_text1.splitlines()]
tokenized_negative1 = [preprocess_text(sentence) for sentence in negative_text1.splitlines()]
tokenized_neutral1 = [preprocess_text(sentence) for sentence in neutral_text1.splitlines()]

all_sentences = tokenized_positive + tokenized_negative + tokenized_neutral +  tokenized_positive1 + tokenized_negative1 + tokenized_neutral1

word2vec_model = Word2Vec(sentences=all_sentences, vector_size=100, window=5, min_count=1, workers=4)
print(word2vec_model)
word2vec_model.save("word2model.save")
t=word2vec_model.wv.most_similar('tệ')
top_n = 500
all_words = list(word2vec_model.wv.index_to_key[:top_n])

# Lấy vectơ tương ứng của mỗi từ
word_vectors = [word2vec_model.wv[word] for word in all_words]

# Giảm chiều dữ liệu từ vectơ 100D xuống 2D bằng PCA
pca = PCA(n_components=2)
result = pca.fit_transform(word_vectors)

# Biểu diễn các từ trên biểu đồ
plt.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(all_words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))

plt.show()
print(t)
print(t)

# # Bước 3: Chuẩn bị dữ liệu cho mô hình phân loại
# X = []
# y = []

# for text in tokenized_positive:
#     X.append([word2vec_model.wv[word] for word in text])
#     y.append('positive')

# for text in tokenized_negative:
#     X.append([word2vec_model.wv[word] for word in text])
#     y.append('negative')

# for text in tokenized_neutral:
#     X.append([word2vec_model.wv[word] for word in text])
#     y.append('neutral')

# X = pad_sequences(X, maxlen=max_length)  # Đảm bảo rằng tất cả các vectơ có cùng kích thước
# y = to_categorical(y)  # Chuyển đổi nhãn thành dạng one-hot encoding

# # Bước 4: Train mô hình phân loại
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model = Sequential()
# model.add(Flatten(input_shape=(max_length, word2vec_model.vector_size)))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(3, activation='softmax'))  # 3 là số lớp (positive, negative, neutral)

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

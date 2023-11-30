import pickle

# Tải mô hình từ file
filename = 'cnn.pkl'
with open(filename, 'rb') as file:
    model = pickle.load(file)

# Dữ liệu văn bản mới cần phân loại
text_samples = [
    "This is a positive review.",
    "I don't like this product.",
    "The movie was great!",
    "The book was boring.",
]
print(model)

# Sử dụng mô hình để dự đoán nhãn của các văn bản mới
predictions = model.predict(text_samples)

# In kết quả dự đoán
for text, label in zip(text_samples, predictions):
    print(f"Văn bản: '{text}'")
    print(f"Nhãn dự đoán: {label}")
    print()
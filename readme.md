// py 3.9>=version>=3.6

b1. đầu tiên là chạy file data.js (mục đích là chia file data ban đầu ra:  khoảng 10k row trong file data_raw thành các file trong folder test_data- train_data)
    Có thể thêm thêm các rows bằng cách copy paste vào trong 6 file trong folder data_raw rồi chạy lại file data.js
b2. Tiền xử lý dữ liệu: Loại bỏ stopword, tách từ tiếng việt, loại bỏ ký tự đặc biệt, vvv rồi lưu vào file pkl 
    chạy file handle.py
b3. trainning: chayj file main.py (với 2 model mà naive bayes và cnn)
    có thể text trựa tiếp khi thay text trong file main.py (text_test) để test model


## Run server
### Flask
1. Install Dependency
```bash
pip install flask pickle sklearn tensorflow gensim flask_cors
```
2. Run flask server
```bash
python server.py
```
### React
1. 
```bash
cd client
```
2. Install dependency 
```bash
npm i
```
3. Run react app
```bash
npm start
```

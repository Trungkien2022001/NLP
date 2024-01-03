const fs = require('fs')
const neu_train = fs.readFileSync('C:\\Users\\84329\\Documents\\NLP\\data_raw\\train_nhan_2.txt', 'utf-8').split('\r\n')
console.log(neu_train.length)
for(i in neu_train){
    fs.writeFileSync(`./train_data/neural/neu_${i}`, neu_train[i])
}

// pos
const pos_train = fs.readFileSync('C:\\Users\\84329\\Documents\\NLP\\data_raw\\train_nhan_1.txt', 'utf-8').split('\r\n')
console.log(pos_train.length)
for(i in pos_train){
    fs.writeFileSync(`./train_data/positive/pos_${i}`, pos_train[i])
}

// neg
const neg_train = fs.readFileSync('C:\\Users\\84329\\Documents\\NLP\\data_raw\\train_nhan_0.txt', 'utf-8').split('\r\n')
console.log(neg_train.length)
for(i in neg_train){
    fs.writeFileSync(`./train_data/negative/neg_${i}`, neg_train[i])
}
const neu = fs.readFileSync('C:\\Users\\84329\\Documents\\NLP\\data_raw\\test_nhan_2.txt', 'utf-8').split('\r\n')

for(i in neu){
    fs.writeFileSync(`./test_data/neural/neu_${i}`, neu[i])
}

// pos
const pos = fs.readFileSync('C:\\Users\\84329\\Documents\\NLP\\data_raw\\test_nhan_1.txt', 'utf-8').split('\r\n')
console.log(pos.length)
for(i in pos){
    fs.writeFileSync(`./test_data/positive/pos_${i}`, pos[i])
}

// neg
const neg = fs.readFileSync('C:\\Users\\84329\\Documents\\NLP\\data_raw\\test_nhan_0.txt', 'utf-8').split('\r\n')
console.log(neg.length)
for(i in neg){
    fs.writeFileSync(`./test_data/negative/neg_${i}`, neg[i])
}
console.log('success')
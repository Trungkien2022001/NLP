import matplotlib.pyplot as plt
from pyvi import ViTokenizer, ViPosTagger # thư viện NLP tiếng Việt
from wordcloud import WordCloud, STOPWORDS


from collections import Counter


def senPerCategory(file_paths):
    num_lines_list = []
    for variable_name, file_path in file_paths.items():
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        num_lines = len(lines)
        num_lines_list.append(num_lines)
        print(f"Number of lines in {variable_name}: {num_lines}")

    categories = list(file_paths.keys())
    plt.pie(num_lines_list, labels=categories, autopct='%1.1f%%', startangle=90)
    plt.axis('equal') 
    plt.show()



def lengthSenPerCategory(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    sentence_lengths = [len(line) for line in text.split('\n')]
    x= list(range(0, len(sentence_lengths)))
    print(round(sum(sentence_lengths) / len(sentence_lengths),2))
    plt.bar(x,sentence_lengths)
    plt.show()

def generateWordCloud(file_path, ):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    stop_words = ['ma', 'anh', 'em', 'vì', 'thế', 'nhưng', 'và' , 'như' ,"sản", "phẩm" , 'là']

    # Generate the word cloud, excluding specified stop words
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stop_words).generate(text)

    # Plot the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    file_paths_test = {
        'negative': './data_raw/test_nhan_0.txt',
        'positive': './data_raw/test_nhan_1.txt',
        'neutral': './data_raw/test_nhan_2.txt',
    }
    file_paths_train = {
        'negative': './data_raw/train_nhan_0.txt',
        'positive': './data_raw/train_nhan_1.txt',
        'neutral': './data_raw/train_nhan_2.txt'
    }
    # senPerCategory(file_paths_test)
    # lengthSenPerCategory('./data_raw/train_nhan_2.txt')
    generateWordCloud('./data_raw/train_nhan_2.txt')
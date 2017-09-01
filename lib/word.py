from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import numpy as np
import jieba
import csv


def trainToken(filePath, numOfData, numOfClass):
    texts = []
    labels = []
    tokenizer = Tokenizer()

    jieba.add_word('建議')
    jieba.add_word('現在')
    jieba.add_word('礦泉水')
    jieba.add_word('==')
    jieba.add_word('嘉義')
    jieba.add_word('學長')
    jieba.add_word('哈囉')
    jieba.add_word('右轉')
    jieba.add_word('第一個')
    jieba.add_word('認領')
    jieba.add_word('上課')
    jieba.add_word('腳踏車')

    with open(filePath, 'r') as f:
        for row in csv.reader(f):
            text = row[numOfData]
            label = row[numOfClass]
            cutWords = jieba.lcut(text)

            texts.append(' '.join(cutWords))
            labels.append(label)

        f.close()

    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    data = pad_sequences(sequences)
    labels = to_categorical(np.array(labels))

    return data, labels

import jieba
import csv

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

data = []

with open('./data/rawwords.csv', 'r') as f:
    for row in csv.reader(f):
        cutWord = jieba.lcut(row[1])
        data.append(cutWord)

    f.close()

chinese = Word2Vec(data)

import jieba
import codecs
import logging
from nltk.corpus import stopwords

logging.basicConfig(level=logging.INFO)  # Logging LEVEL=INFO
jieba.enable_parallel(4)  # Parallel

output = codecs.open('wiki.zhTW.text.cut', 'w', errors='ignore', encoding='utf-8')  # Output file

with codecs.open('wiki.zhTW.text', 'r', errors='ignore', encoding='utf-8') as rawdata:
    index = 0
    for line in rawdata:
        try :
            words = jieba.lcut(line.rstrip('\n'))
            output.write(' '.join(words))
            logging.info('已處理' + str(index+1) + '行')
            index = index + 1
        except:
            logging.info('出現例外狀況')
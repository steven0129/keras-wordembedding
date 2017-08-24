# keras-wordembedding

## Datasets

[維基百科:資料庫下載](https://zh.wikipedia.org/wiki/Wikipedia:%E6%95%B0%E6%8D%AE%E5%BA%93%E4%B8%8B%E8%BD%BD)

## Step By Step

### Step1: preprocess wiki dataset

```python
python wiki-preprocess.py zhwiki-articles.xml.bz2 wiki.zh.text
```

### Step2: transfer simpled chinese into traditinal chinese

```bash
sudo opencc -i wiki.zh.text -o wiki.zhTW.text -c s2t
```

<b style="color:red;">Install Opencc: https://github.com/BYVoid/OpenCC</b>

### Step4: cut sentence into many words

```python
python cut-word.py
```
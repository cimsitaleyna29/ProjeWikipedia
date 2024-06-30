
# (NLP - Text Preprocessing & Text Visualization)

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from textblob import Word, TextBlob
from warnings import filterwarnings


filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 200)

# Datayı okumak
df = pd.read_csv("wiki_data.csv", index_col=0)
df.head()
df = df[:2000]

df.head()
df.shape


'''   
 Adım 1-) Metindeki ön işleme işlemlerini gerçekleştirecek bir fonksiyon yazınız. 
- Büyük küçük harf dönüşümü yapınız.
- Noktalama işaretlerini çıkarınız.
- Numerik ifadeleri çıkarınız.

'''



def clean_text(text_column):
    # Normalizing Case Folding
    text_column = text_column.str.lower()
    # Punctuations
    text_column = text_column.str.replace(r'[^\w\s]', '')
    text_column = text_column.str.replace("\n", '')
    # Numbers
    text_column = text_column.str.replace('\d', '')
    return text_column



df["text"] = clean_text(df["text"])
df.head()


'''
 Adım 2-) Metin içinde öz nitelik çıkarımı yaparken önemli olmayan kelimeleri çıkaracak fonksiyon yazınız.

'''



def remove_stopwords(text_column):
    stop_words = stopwords.words('english')
    text_column = text_column.apply(lambda x: " ".join(x for x in str(x).split() if x not in stop_words))
    return text_column



df["text"] = remove_stopwords(df["text"])


'''
 Adım 3-) Metinde az tekrarlayan kelimeleri bulunuz.

'''

pd.Series(' '.join(df['text']).split()).value_counts()[-1000:]

'''
 Adım 4-) Metinde az tekrarlayan kelimeleri metin içerisinden çıkartınız. ( lambda fonksiyonunu kullanınız.)

'''

sil = pd.Series(' '.join(df['text']).split()).value_counts()[-1000:]
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in sil))

'''
 Adım 5-) : Metinleri tokenize edip sonuçları gözlemleyiniz.

'''
df["text"].apply(lambda x: TextBlob(x).words)

'''
 Adım 6-) Lemmatization işlemini yapınız.
 ran, runs, running -> run (normalleştirme)
'''

df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
df.head()

'''
 Adım 7-) Metindeki terimlerin frekanslarını hesaplayınız. ( Barplot grafiği için gerekli)

'''

tf = df["text"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.head()

'''
 Adım 8-) Barplot grafiğini oluşturunuz.

'''

# Sutünların isimlendirilmesi
tf.columns = ["words", "tf"]

# 5000'den fazla geçen kelimelerin görselleştirilmesi.
tf[tf["tf"] > 2000].plot.bar(x="words", y="tf")
plt.show()

# Kelimeleri WordCloud ile görselleştiriniz.

# Kelimeleri birleştiriniz.
text = " ".join(i for i in df["text"])

# wordcloud görselleştirmenin özelliklerini belirliyoruz.
wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="black").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


'''
Adım 9-) Tüm fonksiyonları tek bir aşama olarak yazınız.
- Metin Ön işleme işlemlerini gerçekleştirniz.
- Görselleştirme işlemlerini fonksiyona argüman olarak ekleyiniz.
- Fonksiyonu açıklayan 'docstring' yazınız.  
 
'''

df = pd.read_csv("wiki_data.csv", index_col=0)



def wiki_preprocess(text_column, Barplot=False, Wordcloud=False):
    """
    Textler üzerinde ön işleme işlemlerini yapar.

    :param text_column: DataFrame'deki textlerin olduğu değişken
    :param Barplot: Barplot görselleştirme
    :param Wordcloud: Wordcloud görselleştirme.
    :return: text_column


    Example:
            wiki_preprocess(dataframe[col_name])

    """
    # Normalizing Case Folding
    text_column = text_column.str.lower()
    # Punctuations
    text_column = text_column.str.replace('[^\w\s]', '')
    text_column = text_column.str.replace("\n", '')
    # Numbers
    text_column = text_column.str.replace('\d', '')
    # Stopwords
    sw = stopwords.words('English')
    text_column = text_column.apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))
    # Rarewords / Custom Words
    sil = pd.Series(' '.join(text_column).split()).value_counts()[-1000:]
    text_column = text_column.apply(lambda x: " ".join(x for x in x.split() if x not in sil))

    if Barplot:
        # Terim Frekanslarının Hesaplanması
        tf = text_column.apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
        # Sutünların isimlendirilmesi
        tf.columns = ["words", "tf"]
        # 5000'den fazla geçen kelimelerin görselleştirilmesi.
        tf[tf["tf"] > 2000].plot.bar(x="words", y="tf")
        plt.show()

    if Wordcloud:
        # Kelimelerin birleştirilmesi.
        text_column = " ".join(i for i in text_column)
        # wordcloud görselleştirmenin özelliklerini belirlenmesi.
        wordcloud = WordCloud(max_font_size=50,
                              max_words=100,
                              background_color="white").generate(text_column)
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()

    return text_column



wiki_preprocess(df["text"])
wiki_preprocess(df["text"], True, True)

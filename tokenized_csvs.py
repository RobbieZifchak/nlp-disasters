import pandas as pd
pd.set_option('display.max_rows', 999)
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string, re
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt
import matplotlib as cm



df = pd.read_csv('train.csv')
testdf = pd.read_csv('test.csv')


df['token'] = df.text.str.replace(r'http\S+', "")
df.location = df.location.str.lower()
testdf['token'] = testdf.text.str.replace(r'http\S+', "")

df
testdf

# setting tokenizer
tokenizer = RegexpTokenizer(r'(?u)(?<![@])#?\b\w\w+\b')

# setting stopwords and punctuations
sw_list = stopwords.words('english')
sw_list += ["''", '""', '...', '``', '’', '“', '’', '”', '‘', '‘', '©' 'airplane mode', 'co', 'http', 'https', 'û_', 'û_https', 'amp']
sw_set = set(sw_list)

# function to tokenize and remove stop words
def process_text(article):
    tokens = tokenizer.tokenize(article)
    stopwords_removed = [token.lower() for token in tokens if token.lower() not in sw_set]
    return stopwords_removed


df.token = list(map(process_text, df.token))
testdf.token = list(map(process_text, testdf.token))

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def lemmatize_text(dataframe):
    lemmatized_review=[]
    for w in dataframe:
        lemmatized_review.append(lemmatizer.lemmatize(w))
    return lemmatized_review

df.token = df.token.apply(lemmatize_text)
testdf.token = testdf.token.apply(lemmatize_text)

df.to_csv('train_with_tokens.csv')
testdf.to_csv('test_with_tokens.csv')

real = df[df['target']==1]
fake = df[df['target']==0]

reallist = real.token.tolist()
fakelist = fake.token.tolist()


reallist[0:5]
fakelist[0:5]

total_vocab_real = set()
for list in reallist:
    total_vocab_real.update(list)
len(total_vocab_real)
total_vocab_real

total_vocab_fake = set()
for list in fakelist:
    total_vocab_fake.update(list)
len(total_vocab_fake)
total_vocab_fake


all_real_tokens = [item for sublist in reallist for item in sublist]
all_fake_tokens = [item for sublist in fakelist for item in sublist]



real_freq = FreqDist(all_real_tokens)
fake_freq = FreqDist(all_fake_tokens)

real_freq.most_common(25)

fake_freq.most_common(25)


real_total_word_count = sum(real_freq.values())
real_top_25 = real_freq.most_common(25)
print("Word \t\t Normalized Frequency")
print()
for word in real_top_25:
    normalized_frequency = word[1]/real_total_word_count
    print("{} \t\t {:.4}".format(word[0], normalized_frequency))

fake_total_word_count = sum(fake_freq.values())
fake_top_25 = fake_freq.most_common(25)
print("Word \t\t Normalized Frequency")
print()
for word in fake_top_25:
    normalized_frequency = word[1]/fake_total_word_count
    print("{} \t\t {:.4}".format(word[0], normalized_frequency))



# create counts of satire and not satire with values and words
real_bar_counts = [x[1] for x in real_freq.most_common(25)]
real_bar_words = [x[0] for x in real_freq.most_common(25)]

fake_bar_counts = [x[1] for x in fake_freq.most_common(25)]
fake_bar_words = [x[0] for x in fake_freq.most_common(25)]


new_figure = plt.figure(figsize=(16,4))

ax = new_figure.add_subplot(121)
ax2 = new_figure.add_subplot(122)

# Generate a line plot on first axes
ax.bar(real_bar_words, real_bar_counts)
# ax.plot(colormap='PRGn')

# Draw a scatter plot on 2nd axes
ax2.bar(fake_bar_words, fake_bar_counts)

ax.title.set_text('Real Disaster')
ax2.title.set_text('Fake Disaster')

for ax in new_figure.axes:
    plt.sca(ax)
    plt.xticks(rotation=60)

plt.tight_layout(pad=0)

# plt.savefig('word count bar graphs.png')

plt.show()


# Getting our data into a dictionary
# FORMAT:  dictionary = dict(zip(keys, values))
# !pip install wordcloud
from wordcloud import WordCloud
real_dictionary = dict(zip(real_bar_words, real_bar_counts))
fake_dictionary = dict(zip(fake_bar_words, fake_bar_counts))


# Create the word cloud:

wordcloud = WordCloud(colormap='Spectral').generate_from_frequencies(real_dictionary)

# Display the generated image w/ matplotlib:

plt.figure(figsize=(7, 7), facecolor='k')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig('realtop25.png')

plt.show()


wordcloud = WordCloud(colormap='Spectral').generate_from_frequencies(fake_dictionary)

plt.figure(figsize=(7, 7), facecolor='k')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig('faketop25.png')

plt.show()


def convert(s):

    # initialization of string to ""
    new = ""

    # traverse in the string
    for x in s:
        new += (x + " ")

    # return string
    return new

real_string = convert(all_real_tokens)
fake_string = convert(all_fake_tokens)



from os import path, getcwd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator


mask = np.array(Image.open("fire_emoji.png"))
wordcloud = WordCloud(mask=mask).generate(real_string)
# create coloring from image
image_colors = ImageColorGenerator(mask)
plt.figure(figsize=[7,7])
plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")
plt.savefig('realtweets.png')
plt.show()


mask = np.array(Image.open("fire_emoji.png"))
wordcloud = WordCloud(mask=mask).generate(fake_string)
# create coloring from image
image_colors = ImageColorGenerator(mask)
plt.figure(figsize=[7,7])
plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")
plt.savefig('faketweets.png')
plt.show()

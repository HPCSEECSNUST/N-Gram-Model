import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import defaultdict

#removing punctuations
#library that contains punctuation
import string
string.punctuation
plt.style.use(style='seaborn')


#removing stopwords
#library that contains stopwords
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


#defining the function to remove punctuation
def remove_punctuation(text):
  if(type(text)==float):
    return text
  ans=""  
  for i in text:     
    if i not in string.punctuation:
      ans+=i    
  return ans


def generate_N_grams(text, ngram):
    words = [word for word in text.split() if word not in set(stopwords.words('english'))]
    temp = zip(*[words[i:] for i in range(ngram)])
    ans = [' '.join(ngram_tuple) for ngram_tuple in temp]
    return ans



#user input
ngram=int(input("What value of n-gram do you want?"))





df=pd.read_csv('all-data.csv',encoding = "ISO-8859-1")
print(df.head())
df.info()
print(df.isna().sum())
print(df['Sentiment'].value_counts())
y=df['Sentiment'].values
print(y.shape)
x=df['News Headline'].values
print(x.shape)
(x_train,x_test,y_train,y_test)=train_test_split(x,y,test_size=0.4)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
df1=pd.DataFrame(x_train)
df1=df1.rename(columns={0:'news'})
df2=pd.DataFrame(y_train)
df2=df2.rename(columns={0:'sentiment'})
df_train=pd.concat([df1,df2],axis=1)
print("Training Dataset")
print(df_train.head())

df3=pd.DataFrame(x_test)
df3=df3.rename(columns={0:'news'})
df4=pd.DataFrame(y_test)
df4=df2.rename(columns={0:'sentiment'})
df_test=pd.concat([df3,df4],axis=1)
print("Testing Dataset")
print(df_test.head())

#storing the puntuation free text in a new column called clean_msg
df_train['news']= df_train['news'].apply(lambda x:remove_punctuation(x))
df_test['news']= df_test['news'].apply(lambda x:remove_punctuation(x))
#punctuations are removed from news column in train dataset)
print("Training Dataset without Punctuations")
print(df_train.head())
#punctuations are removed from news column in test dataset)
print("Testing Dataset without Punctuations")
print(df_test.head())
positiveValues=defaultdict(int)
negativeValues=defaultdict(int)
neutralValues=defaultdict(int)
#get the count of every word in both the columns of df_train and df_test dataframes
#get the count of every word in both the columns of df_train and df_test dataframes where sentiment="positive"
for text in df_train[df_train.sentiment=="positive"].news:
  for word in generate_N_grams(text,ngram):
    positiveValues[word]+=1
#get the count of every word in both the columns of df_train and df_test dataframes where sentiment="negative"
for text in df_train[df_train.sentiment=="negative"].news:
  for word in generate_N_grams(text,ngram):
    negativeValues[word]+=1
#get the count of every word in both the columns of df_train and df_test dataframes where sentiment="neutral"
for text in df_train[df_train.sentiment=="neutral"].news:
  for word in generate_N_grams(text,ngram):
    neutralValues[word]+=1
#focus on more frequently occuring words for every sentiment=>
#sort in DO wrt 2nd column in each of positiveValues,negativeValues and neutralValues
df_positive=pd.DataFrame(sorted(positiveValues.items(),key=lambda x:x[1],reverse=True))
df_negative=pd.DataFrame(sorted(negativeValues.items(),key=lambda x:x[1],reverse=True))
df_neutral=pd.DataFrame(sorted(neutralValues.items(),key=lambda x:x[1],reverse=True))
pd1=df_positive[0][:10]
pd2=df_positive[1][:10]
ned1=df_negative[0][:10]
ned2=df_negative[1][:10]
nud1=df_neutral[0][:10]
nud2=df_neutral[1][:10]

#plotting positive
plt.figure(1,figsize=(16,4))
plt.bar(pd1,pd2, color
         ='green',
        width = 0.4)
plt.xlabel("Words in positive dataframe")
plt.ylabel("Count")
plt.title("Top 10 words in positive dataframe-UNIGRAM ANALYSIS")
plt.savefig("positive-unigram.png")
plt.show()

#plotting negative
plt.figure(1,figsize=(16,4))
plt.bar(ned1,ned2, color ='red',
        width = 0.4)
plt.xlabel("Words in negative dataframe")
plt.ylabel("Count")
plt.title("Top 10 words in negative dataframe-UNIGRAM ANALYSIS")
plt.savefig("negative-unigram.png")
plt.show()

#plotting neutral
plt.figure(1,figsize=(16,4))
plt.bar(nud1,nud2, color ='yellow',
        width = 0.4)
plt.xlabel("Words in neutral dataframe")
plt.ylabel("Count")
plt.title("Top 10 words in neutral dataframe-UNIGRAM ANALYSIS")
plt.savefig("neutral-unigram.png")
plt.show()

# Function to calculate n-gram probabilities
def calculate_ngram_probabilities(ngrams):
    ngram_freq = defaultdict(int)
    context_freq = defaultdict(int)
    
    for ngram in ngrams:
        context = ' '.join(ngram.split()[:-1])
        ngram_freq[ngram] += 1
        context_freq[context] += 1
    
    probabilities = {ngram: freq / context_freq[' '.join(ngram.split()[:-1])] for ngram, freq in ngram_freq.items()}
    return probabilities

# Function to predict the next word
def predict_next_word(context, ngram_probabilities):
    context = ' '.join(context.split()[-(ngram-1):])
    candidates = {ngram.split()[-1]: prob for ngram, prob in ngram_probabilities.items() if ' '.join(ngram.split()[:-1]) == context}
    if not candidates:
        return None
    next_word = max(candidates, key=candidates.get)
    return next_word

# Calculate n-gram probabilities for each sentiment
positive_probabilities = calculate_ngram_probabilities(positiveValues.keys())
negative_probabilities = calculate_ngram_probabilities(negativeValues.keys())
neutral_probabilities = calculate_ngram_probabilities(neutralValues.keys())

# Example of predicting the next word for a given context
context = input("Enter your sentence: ")
next_word_positive = predict_next_word(context, positive_probabilities)
next_word_negative = predict_next_word(context, negative_probabilities)
next_word_neutral = predict_next_word(context, neutral_probabilities)

print(f"Next word prediction for context '{context}' - Positive sentiment: {next_word_positive}")
print(f"Next word prediction for context '{context}' - Negative sentiment: {next_word_negative}")
print(f"Next word prediction for context '{context}' - Neutral sentiment: {next_word_neutral}")





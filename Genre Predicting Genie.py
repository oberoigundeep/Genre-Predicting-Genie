#!/usr/bin/env python
# coding: utf-8

# # GENRE-PREDICTING GENIE

# ### Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import nltk
import re
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV


# ### Importing CSV File

# In[2]:


df_raw = pd.read_csv('./movie_genres.csv')


# In[3]:


df_raw #raw Dataframe


# ### Refining Dataframe

# In[4]:


#Dividing Dataframe into smaller dataframes having relevant Genres
df_horror = df_raw[df_raw.Genre == 'horror' ]
df_comedy = df_raw[df_raw.Genre == 'comedy' ]
df_action = df_raw[df_raw.Genre == 'action' ]
df_adventure = df_raw[df_raw.Genre == 'adventure' ]
df_crime = df_raw[df_raw.Genre == 'crime' ]
df_mystery = df_raw[df_raw.Genre == 'mystery' ]
df_fantasy = df_raw[df_raw.Genre == 'fantasy' ]
df_historical = df_raw[df_raw.Genre == 'historical' ]
df_romance = df_raw[df_raw.Genre == 'romance' ]
df_sf = df_raw[df_raw.Genre == 'science fiction' ]
df_thriller = df_raw[df_raw.Genre == 'thriller' ]
df_western = df_raw[df_raw.Genre == 'western' ]
df_suspense = df_raw[df_raw.Genre == 'suspense' ]
#Combinig above dataframes
df_submain = pd.concat([df_horror, df_comedy, df_action, df_adventure, df_crime, df_mystery, df_fantasy, df_historical, df_romance, df_sf, df_thriller, df_western, df_suspense], axis = 0)


# In[5]:


df_submain


# In[6]:


df_main = df_submain[['Title', 'Genre']] # creting main datframe with required feilds


# In[7]:


df_main


# In[8]:


total_genres = set(df_main.Genre.values)
print(len(total_genres))


# In[9]:


genre = {}
g_counter = 0
for g in total_genres:
    genre[g] = g_counter
    g_counter += 1    


# In[10]:


genre


# In[11]:


df_main['num_genre'] = df_main.Genre.map(genre)


# ### Creating Bag of Words

# In[12]:


def extract_words(sentence):
    '''This is to clean and tokenize words'''
    ignore_words = ['a', 'the', 'if', 'br', 'and', 'of', 'to', 'is', 'are', 'he', 'she', 'my', 'you', 'it','how']
    words = re.sub("[^\w]", " ",  sentence).split() # this replaces all special chars with ' '
    words = [word.lower() for word in words]
    words_cleaned = [w.lower() for w in words if w not in ignore_words]
    return words_cleaned 


def map_book(hash_map, tokens):
    if tokens is not None:
        for word in tokens:
            # Word Exist?
            if word in hash_map:
                hash_map[word] = hash_map[word] + 1
            else:
                hash_map[word] = 1

        return hash_map
    else:
        return None
    

def make_hash_map(df):
    hash_map = {}
    for index, row in df.iterrows():
        hash_map = map_book(hash_map, extract_words(row['Title']))
    return hash_map


hash_map = make_hash_map(df_main)


# In[13]:


def frequent_vocab(word_freq, max_features): 
    counter = 0  #initialize counter with the value zero
    vocab = []   # create an empty list called vocab
    # list words in the dictionary in descending order of frequency
    for key, value in sorted(word_freq.items(), key=lambda item: (item[1], item[0]), reverse=True): 
       #loop function to get the top (max_features) number of words
        if counter<max_features: 
            vocab.append(key)
            counter+=1
        else: break
    return vocab


# In[14]:


vocabulary = frequent_vocab(hash_map, 5000)


# In[15]:


vocabulary


# In[16]:


def bagofwords(sentence, words):
    sentence_words = extract_words(sentence) #tokenize sentences/ tweets and assign it to variable sentence_words
    # frequency word count
    bag = np.zeros(len(words)) #create a NumPy array made up of zeroes with size len(words)
    # loop through data and add value of 1 when token is present in the tweet
    for sw in sentence_words:
        for i,word in enumerate(words):
            if word == sw: 
                bag[i] += 1
                
    return np.array(bag) # return the bag of word for one tweet


# In[17]:


n_vocab = len(vocabulary)
n_titles = len(df_main)
bag_of_words = np.zeros([n_titles,n_vocab])
# use loop function to add new row for each tweet. 
for ii in range(n_titles): 
    #call out the previous function 'bagofwords'. see the inputs: sentence and words
    bag_of_words[ii,:] = bagofwords(df_main['Title'].iloc[ii], vocabulary) 


# In[18]:


bag_of_words


# ### Total Frequency, Inverse Document Frequency

# In[19]:


#tfidf:
#idf
numtitles, numvocab = np.shape(bag_of_words)
N = numtitles
word_frequency = np.empty(numvocab)
for word in range(numvocab):
    word_frequency[word]=np.sum((bag_of_words[:,word]>0)) #total no. of documents containing the words
idf = np.log(N/word_frequency)


# In[20]:


idf.shape


# In[21]:


tfidf = np.empty([numtitles, numvocab])

#loop through the tweets, multiply term frequency (represented by bag of words) with idf
for doc in range(numtitles):
    tfidf[doc, :]=bag_of_words[doc, :]*idf


# In[22]:


tfidf.shape


# ## Training our "Genie"

# In[23]:


X_train,X_test,Y_train,Y_test = train_test_split(tfidf,df_main['num_genre'].values,shuffle=True)


# In[24]:


print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape, sep = '\n')


# In[25]:


def classify(rf, X_all, Y_all): #Take in the untrained model, tfidf array, and the values from the training target.
    X_train,X_test,Y_train,Y_test = train_test_split(X_all,Y_all,shuffle=True) #Randomly split the two into a train and test set
    logreg.fit(X_train,Y_train) #Fit the model on the training set
    print(rf.score(X_test,Y_test)) #Print the score on the test set
    return logreg #Return the trained model.


# In[26]:


logreg = LogisticRegression( multi_class = 'multinomial', solver = 'lbfgs')
#logreg = LogisticRegression()

X_all = tfidf
Y_all = df_main['num_genre'].values
logreg = classify(logreg, X_all, Y_all)


# In[27]:


# Define your hyperparameters here
#parameters = {'C':[0.001, 0.01, 0.1, 1, 10], 'tol':[0.0001, 0.001, 0.01], 'max_iter':[100, 1000]}

#clf = GridSearchCV(logreg, parameters, cv=3, return_train_score=True)
#clf.fit(X_all, Y_all)
#print(clf.best_params_, clf.best_score_)


# In[28]:


genre_r = {}
for a in genre:
    genre_r[genre[a]] = a
genre_r


# In[29]:


def genre_predictor(title):
    word_vector = bagofwords(title, vocabulary)
    word_tfidf = word_vector*idf
    prediction = logreg.predict(word_tfidf.reshape(1, -1)) #predict wether a tweet is relevant or not relevant to natural disaster
    results = genre_r #creating a set containing the potential results. You can change the 'Relevant' and 'Not relevant' tag
    return results[int(prediction)]


# ## Designing Interactive Genie

# In[30]:


#Designing Genie


# In[31]:


GREETING_INPUTS = ["hello", "hi", "greetings", "sup", "what's up","hey", "hey there",'sup', 'yo', 'heyo']
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me",'hey-o!', 'howdy']
GENIE_MAGIC = ["!Avada Kedavra!", '!Expecto patronum!', '!abracadabra!', '!Hocus Pocus!', '!bibbidi-bobbidi-boo!']


# In[32]:


def greeting(sentence):
    for word in sentence.split(): # Looks at each word in your sentence
        if word.lower() in GREETING_INPUTS: # checks if the word matches a GREETING_INPUT
            return random.choice(GREETING_RESPONSES) # replies with a GREETING_RESPONSE


# In[33]:


flag = True
print("GENIE: Welcome to the world of movies! I am your Genie!!!")
while flag == True:
    print("GENIE: Master, Please enter the movie title you wish to search for")
    user_title = input()
    user_title = user_title.lower()
    if user_title not in GREETING_INPUTS and user_title not in ['bye', 'thank you', 'goodbye']:
        i_genre = genre_predictor(user_title)
        print(random.choice(GENIE_MAGIC))
        print("GENIE: The movie title belongs to the Genre \t\"",i_genre.upper(), '"' )
        print("\n\n","GENIE: Here are 5 titles related to the previous movie:")
        if i_genre == 'adventure':
            ex = df_adventure.head(5)
        elif i_genre == 'romance':
            ex = df_romance.head(5)
        elif i_genre == 'mystery':
            ex = df_mystery.head(5)
        elif i_genre == 'action':
            ex = df_action.head(5)
        elif i_genre == 'western':
            ex = df_western.head(5)
        elif i_genre == 'comedy':
            ex = df_comedy.head(5)
        elif i_genre == 'fantasy':
            ex = df_fantasy.head(5)
        elif i_genre == 'thriller':
            ex = df_thriller.head(5)
        elif i_genre == 'horror':
            ex = df_horror.head(5)
        elif i_genre == 'suspense':
            ex = df_suspense.head(5)
        elif i_genre == 'crime':
            ex = df_crime.head(5)
        elif i_genre == 'historical':
            ex = df_historical.head(5)
        elif i_genre == 'science fiction':
            ex = df_sf.head(5)
        print(ex['Title'], end = '\n\n\n')
    elif user_title in GREETING_INPUTS:
        greet = greeting(user_title)
        print("GENIE:", greet, 'Master, Your wish is my command')
    elif user_title in ['bye', 'thank you', 'goodbye']:
        print("GENIE: Good-Bye! See you later! ")
        print("*Genie goes back to lamp*")
        flag = False


# # GENIE BECOMES A REALITY...!!

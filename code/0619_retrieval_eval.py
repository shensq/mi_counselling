import glob
from nltk.tokenize import word_tokenize
import re
import numpy as np
from tqdm import tqdm
import pickle

filepath = '/Users/shensq/Documents/NLP/MI_data/datasetMI_real_standardized/annotations/'
files = glob.glob(filepath+'[1-9m]*.txt')

from gensim.models import KeyedVectors
word2vec = KeyedVectors.load_word2vec_format('/Users/shensq/Documents/NLP/GoogleNews-vectors-negative300.bin', binary=True)  
print("Word2vec loaded")

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


def clean_text(text):
    text = text.lower()
    text = re.sub("it's", "it is", text)
    text = re.sub("i'm", "i am", text)
    text = re.sub("he's", "he is", text)
    text = re.sub("she's", "she is", text)
    text = re.sub("that's", "that is", text)
    text = re.sub("what's", "what is", text)
    text = re.sub("where's", "where is", text)
    text = re.sub("he's", "he is", text)
    text = re.sub("\'s", " \'s",text)
    text = re.sub("\'ll", " will", text)
    text = re.sub("\'ve", " have", text)
    text = re.sub("\'d", " would", text)
    text = re.sub("\'re", " are", text)
    text = re.sub("don't", "do not", text)
    text = re.sub("won't", "will not", text)
    text = re.sub("can't", "can not", text)
    text = re.sub("[-()\"#/@;:<>{}+=~.…,|!?]", "", text)
    return text

def check_size(xy,min_length,max_length,num_turns=5):
    xs = xy[0]
    y = xy[1]
    if len(xs)!=num_turns:
        return False
    if len(y)<=min_length or len(y)>=max_length:
        return False
    for x in xs:
        if len(x)>=max_length: # no requirement for minimum length in inputs
            return False
    return True

num_turns = 5
x_all = []
y_all = []

code_set = set([ 'CR',  'SR', 'GIV', 'QUEST', 'SEEK', 'AF', 'EMPH', 'PWOP', 'PWP', 'CON'])
for file in files:
    f = open(file)
    data = []
    for line in f: 
        line = line.split() # split only with space 
        data.append(line) # data[i] is a list of tokens in a sentence
    data.reverse() # reverse the order for sentences, which makes it easier to trace back
        
    for i,sen in enumerate(data):
        if sen[0]=='SR' or sen[0]=='CR':
            # these are used to keep a single input pair
            y = sen[1:]
            x = []
        
            pointer = i
#             while data[pointer][0]!='T:' and data[pointer][0]!='T.':
            while data[pointer][0]!='T:':
                pointer+=1 

            # now the data[pointer] is the "T" turn which contains the reflection
            # trace back k valid turns
            for j in range(pointer+1,len(data)):
                if data[j][0] in code_set:
                    continue # skip sentence with MI codes
                x.append(data[j][1:]) # add an utterance into y
                if len(x)>=num_turns:
                    break # if reach the expected turn, stop 
            x.reverse() # reverse the sentence order back to normal
            x_all.append(x)
            y_all.append(y)

xy_filter = [xy for xy in zip(x_all,y_all) if check_size(xy,1,100,num_turns)]



#===============
doc_utterances = [] # [doc][sen]
doc_responses = [] # [doc][sen]

code_set = set([ 'CR',  'SR', 'GIV', 'QUEST', 'SEEK', 'AF', 'EMPH', 'PWOP', 'PWP', 'CON'])

#=====get utterance& responses list (untokenized) ========
for file in files:
    f = open(file)
    data = []
    for line in f: 
        line = line.split() # split only with space 
        data.append(line) # data[i] is a list of tokens in a sentence
    
    data_utterance = []
    data_response = []
    for i,sen in enumerate(data):
        if sen[0]=='SR' or sen[0]=='CR':
            data_response.append(' '.join(sen[1:])) # untokenized response
        if sen[0] not in code_set:
            data_utterance.append(' '.join(sen[1:])) # skip line with code to avoid duplicate
    doc_responses.append(data_response)
    doc_utterances.append(data_utterance)


# ================ Get word2index & tokenized utterance list =============
word2index = {}
index2word = []

doc_utterances_tokenized = [] # [doc][sen] -> list of tokens

for doc_id,doc in enumerate(doc_utterances):
    doc_tokenized = []
    for sen in doc:
        sen = clean_text(sen)
        sen = sen.split()
        doc_tokenized.append(sen)
        for word in sen:
            if word not in word2index:
                word2index[word]=len(word2index)
                index2word.append(word)
    doc_utterances_tokenized.append(doc_tokenized)    
    
doc_utterances_tokenized_flat = []
for doc in doc_utterances_tokenized:
    doc_utterances_tokenized_flat.append([w for sen in doc for w in sen])

# ========= Get TF-IDF ============
tf = np.zeros([len(word2index),len(files)]) # [word][doc] -> term frequency

for i,doc in enumerate(doc_utterances_tokenized):
    for sen in doc:
        for word in sen:
            tf[word2index[word],i]+=1

# Inverse document frequency, count how many documents does each word appear in.
idf = np.zeros(len(word2index)) # [doc] -> inverse document frequency
df = np.sum(np.where(tf!=0,np.ones(tf.shape),np.zeros(tf.shape)),axis=1)
idf = np.log10(len(word2index)/df).reshape(len(word2index),1)
tf_idf = tf*idf

# normalize vector for each document
for i in range(tf_idf.shape[1]):
    tf_idf[:,i]=tf_idf[:,i]/np.linalg.norm(tf_idf[:,i])


# ====== tokenized repsonse list=======
doc_responses_tokenized = [] # [doc][sen] -> list of tokens
doc_responses_embedding = [] # [doc][sen] -> (300,)array

for doc_id,doc in enumerate(doc_responses):
    doc_tokenized = []
    doc_embedding = []
    for sen in doc:
        sen = clean_text(sen)
        sen = sen.split()
        doc_tokenized.append(sen)
        
        filter_sen = [w for w in sen if w not in stop_words and w in word2vec]
        sen_embedding = np.zeros(300)
        for word in filter_sen:
            sen_embedding += word2vec[word]
        if len(filter_sen):
            sen_embedding/=len(filter_sen)
        doc_embedding.append(sen_embedding)
    doc_responses_tokenized.append(doc_tokenized)   
    doc_responses_embedding.append(doc_embedding)


counter = 0
memory_flat = []
for x,y in tqdm(xy_filter):
    # if counter>0:
    #     break
    # counter+=1
    # for a single x sample
    x_concat = [] # tokens of k-utternaces
    x_filtered = [] # used for word2vec embedding 
    for sen in x:
        sen = ' '.join(sen).lower()
        sen = clean_text(sen)
        sen = sen.split()
        for w in sen:
            x_concat.append(w)
            if w in word2vec and w not in stop_words:
                x_filtered.append(w)
    # ==== get embedding & tfidf & tokens list for input ====
    query_embedding = np.zeros(300)
    query_tfidf = np.zeros(len(word2index))

    for w in x_concat:
        query_tfidf[word2index[w]]+=1
    query_tfidf = query_tfidf.reshape(len(word2index),1)*idf # (vocab,1)

    for w in x_filtered:
        query_embedding += word2vec[w]
    query_embedding/=len(x_filtered)

    # Document matching with tf-idf
    score = tf_idf.T.dot(query_tfidf).reshape(len(files))
    top_k_idx = np.argsort(-score)[:5] # top-5 files, can be extended

    # # Document matching WMD
    # WMD_score = [0]*len(files)
    # for i,doc in tqdm(enumerate(doc_utterances_tokenized_flat)):
    #     WMD_score[i] = word2vec.wmdistance(doc, x_concat)

    # # Sentence matching with word2vec
    score_mapping = []
    for idx in top_k_idx:
        sen_scores = np.array(doc_responses_embedding[idx]).dot(query_embedding)
        score_mapping += [(score,idx,sen_idx) for sen_idx,score in enumerate(list(sen_scores))]
    score_mapping.sort(reverse=True) # choose from all the utterances from top-k 

    # Sentence matching with WMD
    # score_mapping = []
    # for idx in top_k_idx:
    #     for sen_idx,sen in enumerate(doc_responses_tokenized[idx]):
    #         sen_score = word2vec.wmdistance(sen, x_concat)
    #         score_mapping.append((sen_score,idx,sen_idx))
    # score_mapping.sort(reverse=True)

    # top_k_sentences = score_mapping[:5]

    # for score,doc_idx,sen_idx in top_k_sentences:
    #     memory_flat.append(doc_responses[doc_idx][sen_idx])


handler = open('multiturns_data/memory_flat','wb')
pickle.dump(memory_flat,handler)
            


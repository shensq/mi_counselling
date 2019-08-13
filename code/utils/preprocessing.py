import glob
import re
import numpy as np
import time
import pickle 
from tqdm import tqdm

def annotate_topic(preprocessed_data_path='./',file_path = '/Users/shensq/Documents/NLP/MI_data/datasetMI_real_standardized/annotations/'):
    """
    Do annotation of topic manually.
    """
    files = glob.glob(file_path+'[1-9m]*.txt') # bad session is exclued
    tag_set = ['Medication adherence','Smoking cessation','Weight management','others']
    file_to_tag=[]
    finished_file = []
    for file in files:
        if file in finished_file:
            continue
        print('='*20+str(len(finished_file)))
        f = open(file)
    #     print(f.readlines()[])

        context = ''
        for line in f.readlines()[:30]:
            context += line
        print(context)
        
        tag = input()
        file_to_tag.append([file,tag]) # in integer. add i to avoid misplace
        finished_file.append(file)
    # convert the index to acctual tag.
    file_to_tag_mod={}
    for file,tag in file_to_tag:
        file_to_tag_mod[file[len(file_path):]]=tag_set[int(tag)]

    with open(preprocessed_data_path+'session_topic','wb') as f:
        pickle.dump(file_to_tag_mod,f)

def parse_text(num_turns=5,preprocessed_data_path='./',file_path = '/Users/shensq/Documents/NLP/MI_data/datasetMI_real_standardized/annotations/'):
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
    files = glob.glob(file_path+'[1-9m]*.txt')
    x_all = []
    y_all = []
    meta_all = []
    code_set = set([ 'CR',  'SR', 'GIV', 'QUEST', 'SEEK', 'AF', 'EMPH', 'PWOP', 'PWP', 'CON']) # all the MITI codes
    pickle_handler = preprocessed_data_path+'session_topic'
    file_to_tag = pickle.load(pickle_handler)

    for file in tqdm(files):
        topic = file_to_tag[file[len(file_path):]]
        f = open(file)
        data = []
        for line in f: 
            line = line.split() # split only with space 
            data.append(line) # data[i] is a list of tokens from a single sentence
        data.reverse() # reverse the order for sentences, which makes it easier to trace back

        for i,sen in enumerate(data):
            if sen[0]=='SR' or sen[0]=='CR':
                code = sen[0]
                y = sen[1:] # get rid of the code itself
                x = []
                pointer = i
                while data[pointer][0]!='T:': # find data[pointer] which contains the copy of detected relection
                    pointer+=1              
                for j in range(pointer+1,len(data)): # trace up k valid utterances from the previous sentences to the top 
                    if data[j][0] in code_set:
                        continue # skip sentence with MI codes since there will be a duplicate copy, and a single sentence may contains multiple codes
                    x.append(data[j][1:]) # add an utterance into x, without the first token  
                    if len(x)>=num_turns:
                        break # if reach the expected turn, stop 
                x.reverse() # reverse the sentence order back to normal
                x_all.append(x)
                y_all.append(y)
                meta_all.append([file[len(file_path):],topic,code])
    xy_filter = [xy for xy in zip(x_all,y_all,meta_all) if check_size(xy,1,100,num_turns)]
    x_all_join = []
    y_all_join = []
    meta_all_join = []
    for xy in xy_filter:
        x_all_join.append([' '.join(i) for i in xy[0]])
        y_all_join.append(' '.join(xy[1]))
        meta_all_join.append(xy[2])
    x_y_meta = list(zip(x_all_join,y_all_join,meta_all_join))
    with open(preprocessed_data_path+'x_y_meta','wb') as f:
        pickle.dump(x_y_meta,f)

def text_standardize(text):
    """
    fixes some issues the spacy tokenizer had on books corpus
    also does some whitespace standardization
    """
    text = text.replace('—', '-')
    text = text.replace('–', '-')
    text = text.replace('―', '-')
    text = text.replace('…', '...')
    text = text.replace('´', "'")
    text = text.replace('’',"'")
    text = text.replace('‘',"")
    text = text.replace('”',"\"")
    text = text.replace('“',"\"")
    text = re.sub(r'''(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)''', r' \1 ', text)
    text = re.sub(r'\s*\n\s*', ' \n ', text)
    text = re.sub(r'[^\S\n]+', ' ', text)
    return text.strip()

def clean_text(text):
    text = text.lower()
    text = re.sub("[’]","\'",text)
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
    text = re.sub("\'re", " are", text)
    text = re.sub("\'d", " would", text)
    text = re.sub("\'re", " are", text)
    text = re.sub("don't", "do not", text)
    text = re.sub("won't", "will not", text)
    text = re.sub("can't", "can not", text)
    text = re.sub("[-()\"#/@;:<>{}+=~.…,|!?]", "", text)
    return text

def main():
    print("Use the preprocessing by importing functions instead.")

if __name__ == "__main__":
    pass
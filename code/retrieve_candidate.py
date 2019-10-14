# get tf-idf info for each documents
# get relevant sentences to be evaluated 
# load the model
# calculate score for every candidates
# pick the one with highest score 
# save the new input
import sys

sys.path.insert(0, '/home/shensq/LIT/pip_package')
import argparse
import glob
import re
import numpy as np
from tqdm import tqdm
import torch
from utils import text_standardize
from pytorch_transformers import GPT2Tokenizer
from gpt_loader import GptDataset, collate_fn, collate_fn_nli, GptDataset_nli, SnliDataset
from torch.utils.data import Dataset, DataLoader
from model import GPT2ClassHeadsModel
import pickle
import logging


def get_doc_utterance(files):
    num_turns = 5
    doc_utterances = []  # [doc][sen]
    doc_responses = []  # [doc][sen]

    code_set = set(['CR', 'SR', 'GIV', 'QUEST', 'SEEK', 'AF', 'EMPH', 'PWOP', 'PWP', 'CON'])

    # =====get utterance& responses list (untokenized) ========
    for file in files:
        f = open(file)
        data = []
        for line in f:
            line = line.split()  # split only with space
            data.append(line)  # data[i] is a list of tokens in a sentence

        data_utterance = []
        data_response = []
        for i, sen in enumerate(data):
            if sen[0] == 'SR' or sen[0] == 'CR':
                data_response.append(' '.join(sen[1:]))  # untokenized response
            if sen[0] not in code_set:
                data_utterance.append(' '.join(sen[1:]))  # skip line with code to avoid duplicate
        doc_responses.append(data_response)
        doc_utterances.append(data_utterance)
    return doc_responses, doc_utterances


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
    text = re.sub("\'s", " \'s", text)
    text = re.sub("\'ll", " will", text)
    text = re.sub("\'ve", " have", text)
    text = re.sub("\'d", " would", text)
    text = re.sub("\'re", " are", text)
    text = re.sub("don't", "do not", text)
    text = re.sub("won't", "will not", text)
    text = re.sub("can't", "can not", text)
    text = re.sub("[-()\"#/@;:<>{}+=~.…,|!?]", "", text)
    return text


def get_tfidf(files, doc_utterances):
    # ================ Get word2index & tokenized utterance list =============
    word2index = {}
    index2word = []

    doc_utterances_tokenized = []  # [doc][sen] -> list of tokens

    for doc_id, doc in enumerate(doc_utterances):
        doc_tokenized = []
        for sen in doc:
            sen = clean_text(sen)
            sen = sen.split()
            doc_tokenized.append(sen)
            for word in sen:
                if word not in word2index:
                    word2index[word] = len(word2index)
                    index2word.append(word)
        doc_utterances_tokenized.append(doc_tokenized)

    doc_utterances_tokenized_flat = []
    for doc in doc_utterances_tokenized:
        doc_utterances_tokenized_flat.append([w for sen in doc for w in sen])

    # ========= Get TF-IDF ============
    tf = np.zeros([len(word2index), len(files)])  # [word][doc] -> term frequency

    for i, doc in enumerate(doc_utterances_tokenized):
        for sen in doc:
            for word in sen:
                tf[word2index[word], i] += 1

    # Inverse document frequency, count how many documents does each word appear in.
    idf = np.zeros(len(word2index))  # [doc] -> inverse document frequency
    df = np.sum(np.where(tf != 0, np.ones(tf.shape), np.zeros(tf.shape)), axis=1)
    idf = np.log10(len(word2index) / df).reshape(len(word2index), 1)
    tf_idf = tf * idf

    # normalize vector for each document
    for i in range(tf_idf.shape[1]):
        tf_idf[:, i] = tf_idf[:, i] / np.linalg.norm(tf_idf[:, i])
    return tf_idf, tf, idf, word2index, index2word


def get_sentence_tfidf(x, word2index, idf):
    x_concat = []  # tokens of k-utternaces
    for sen in x:
        sen = sen.lower()
        sen = clean_text(sen)
        sen = sen.split()
        for w in sen:
            x_concat.append(w)
    query_tfidf = np.zeros(len(word2index))
    for w in x_concat:
        query_tfidf[word2index[w]] += 1
    query_tfidf = query_tfidf.reshape(len(word2index), 1) * idf  # (vocab,1)

    return query_tfidf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default='345M_Alex', type=str, required=False,
                        help="The directory of the model to be tuned.")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--keyword', action='store_true')
    parser.add_argument('--special_input', type=str, default='x_y_meta')
    parser.add_argument('--first_K_tokens', type=int, default=1024)
    # parser.add_argument('--num_turns', type=int, default=5)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    filepath = '../data/datasetMI_real_standardized/annotations/'
    files = glob.glob(filepath + '[1-9m]*.txt')
    model_dir = '../models/' + args.model_dir
    model = GPT2ClassHeadsModel.from_pretrained(model_dir)
    # model = GPT2ClassHeadsModel.from_pretrained('gpt2')
    if torch.cuda.is_available():
        model.cuda()
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    print('Model loaded.')

    pickle_handler = open('../data_processed/x_y_meta_10turn', 'rb')
    x_y_meta = pickle.load(pickle_handler)
    gpt_data = GptDataset_nli(x_y_meta, tokenizer, augment=False, num_turns=10)

    doc_responses, doc_utterances = get_doc_utterance(files)
    tf_idf, tf, idf, word2index, index2word = get_tfidf(files, doc_utterances)

    x_y_meta_aug = []
    for x, y, meta in tqdm(x_y_meta):
        query_tfidf = get_sentence_tfidf(x, word2index, idf)
        doc_score = tf_idf.T.dot(query_tfidf).reshape(len(files))
        top_k_idx = np.argsort(-doc_score)[0]  # pick only one doc
        response_candidates = doc_responses[top_k_idx]

        candidate_score = []
        candidates = list(zip([x] * len(response_candidates), response_candidates, [0] * len(response_candidates)))
        gpt_data.x_encoded, gpt_data.y_encoded, gpt_data.label = gpt_data._split(candidates)
        data_loader = DataLoader(dataset=gpt_data, batch_size=1, shuffle=False, drop_last=False,
                                 collate_fn=collate_fn_nli)
        for token_x, type_x, pos_x, lm_x, label in data_loader:
            if token_x.shape[1] >= 512:
                candidate_score.append(float('-inf'))
                continue
            loss, logits = model(token_x, position_ids=pos_x, token_type_ids=type_x, labels=label)  # [batch,class]
            # candidate_score.append(logits[:, 1].item())  # does not support batch
            candidate_score.append(torch.softmax(logits, 1)[:, 1].item())
        if len(candidate_score) > 0:
            y_aug = response_candidates[np.argmax(candidate_score)]
            x_y_meta_aug.append([x, y, meta, y_aug])
    with open('../data_processed/x_y_meta_aug', 'wb') as f:
        pickle.dump(x_y_meta_aug, f)


if __name__ == "__main__":
    main()



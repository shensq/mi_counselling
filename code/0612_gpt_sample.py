#!/usr/bin/env python3

import argparse
import logging
import pickle
import re
from tqdm import trange
import random 
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
from tqdm import tqdm, trange
from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer
from rouge import Rouge 

USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

#============Duplicate code ==============

# ==== Code for data loading =====
class GptDataset(Dataset):
    # need 3 special tokens
    # # as <ref start> 2
    # $ as <speaker1> 3
    # % as <speaker2> 4
    # '<|endoftext|>' as <eos> 50256
    def __init__(self,x_encoded,y_encoded,num_turns=5):
        self.x_encoded = x_encoded
        self.y_encoded = y_encoded

        self.num_turns = num_turns
        
    def __getitem__(self,index):
        type_x = []
        x = []
        lm_x = []

        ref_start, speaker1,speaker2,eos = 2,3,4,50256
        x += [speaker1] + self.x_encoded[index*self.num_turns]
        type_x += [speaker1]*(len(self.x_encoded[index*self.num_turns])+1)
        
        x += [speaker2] + self.x_encoded[index*self.num_turns+1]
        type_x += [speaker2]*(len(self.x_encoded[index*self.num_turns+1])+1)
        
        x += [speaker1] + self.x_encoded[index*self.num_turns+2]
        type_x += [speaker1]*(len(self.x_encoded[index*self.num_turns+2])+1)
        
        x += [speaker2] + self.x_encoded[index*self.num_turns+3]
        type_x += [speaker2]*(len(self.x_encoded[index*self.num_turns+3])+1)
        
        x += [speaker1] + self.x_encoded[index*self.num_turns+4]
        type_x += [speaker1]*(len(self.x_encoded[index*self.num_turns+4])+1)
        lm_x += [-1]*len(x)
        
        total_input_length = len(x)
        
        x += [ref_start] + self.y_encoded[index] + [50256]
        type_x += [ref_start]*(len(self.y_encoded[index])+2)
        lm_x += [-1] + self.y_encoded[index] + [-1]
        
        position_x = list(range(len(x)))
        
        x = torch.Tensor(x)
        type_x = torch.Tensor(type_x)
        position_x = torch.Tensor(position_x)
        lm_x = torch.Tensor(lm_x)
        x_len = x.shape[0]
        
        return x,type_x,position_x,lm_x,total_input_length

    def __len__(self):
        return len(self.y_encoded)

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq).
    We should build a custom collate_fn rather than using default collate_fn,
    because merging sequences (including padding) is not supported in default.
    Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).
    Args:
        data: list of tuple (src_seq, trg_seq).
            - src_seq: torch tensor of shape (?); variable length.
            - trg_seq: torch tensor of shape (?); variable length.
    Returns:
        src_seqs: torch tensor of shape (batch_size, padded_length).
        src_lengths: list of length (batch_size); valid length for each padded source sequence.
        trg_seqs: torch tensor of shape (batch_size, padded_length).
        trg_lengths: list of length (batch_size); valid length for each padded target sequence.
    """
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # seperate source and target sequences
    src_seqs, trg_seqs, pos_seqs,lm_seqs = zip(*data)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lengths = merge(src_seqs)
    trg_seqs, trg_lengths = merge(trg_seqs)
    pos_seqs, pos_lengths = merge(pos_seqs)
    lm_seqs, lm_lengths = merge(lm_seqs)
    if USE_CUDA:
        src_seqs = src_seqs.cuda()
        trg_seqs = trg_seqs.cuda()
        pos_seqs = pos_seqs.cuda()
        lm_seqs = lm_seqs.cuda()
    return Variable(LongTensor(src_seqs)), Variable(LongTensor(trg_seqs)), Variable(LongTensor(pos_seqs)),Variable(LongTensor(lm_seqs)),  src_lengths

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
    return text

#=========================================


def top_k_logits(logits, k):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)

def sample_sequence(model, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, device='cuda', sample=True):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
        context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)
    prev = context
    output = context
    past = None
    with torch.no_grad():
        for i in trange(length):
            logits, past = model(prev, past=past)
            logits = logits[:, -1, :] / temperature
            logits = top_k_logits(logits, k=top_k)
            log_probs = F.softmax(logits, dim=-1)
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)
            output = torch.cat((output, prev), dim=1)
    return output

def run_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='mi_tuned_5epo', help='pretrained model name or path to local checkpoint')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument('--unconditional', action='store_true', help='If true, unconditional generation.')
    parser.add_argument('--output_dir',type=str,default='generate', help="The name of the output file.")
    args = parser.parse_args()
    print(args)

    if args.batch_size == -1:
        args.batch_size = 1
    assert args.nsamples % args.batch_size == 0

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  === prepare data and model
    # ====== Load GPT2 model ========
    # TODO: use the new data, store to another file
    model_dir = '../gpt2/models/'+args.model_dir
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    # if USE_CUDA:
    #     model.cuda()
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)

    # =============== Load & process data ==============
    # folder = '/home/shensq/gpt_tuning/multiturns_data/'
    folder = '../data_processed/'
    x_flat = pickle.load(open(folder+'x_flat','rb'))
    y_all_join = pickle.load(open(folder+'y_all_join','rb'))

    x_encoded = [tokenizer.encode(text_standardize(x)) for x in x_flat]
    y_encoded = [tokenizer.encode(text_standardize(y)) for y in y_all_join]
    # y_cleaned = [clean_text(y) for y in y_all_join]
    # y_encoded = [tokenizer.convert_tokens_to_ids(y.split()) for y in y_cleaned]

    # x_cleaned = [clean_text(x) for x in x_flat]
    # x_encoded = [tokenizer.convert_tokens_to_ids(x.split()) for x in x_cleaned]

    gpt_data = GptDataset(x_encoded,y_encoded)
    test_size  = int(len(gpt_data)*0.05)
    val_size = int(len(gpt_data)*0.05)
    gpt_train,gpt_test,gpt_val = torch.utils.data.random_split(gpt_data,[len(gpt_data)-test_size-val_size,test_size,val_size])

    # data_loader = DataLoader(dataset=gpt_train,batch_size=args.train_batch_size,shuffle=True,drop_last=True,collate_fn=collate_fn)
    test_loader = DataLoader(dataset=gpt_test,batch_size=args.batch_size,shuffle=True,drop_last=True)
# ====

    model.eval()

    if args.length == -1:
        args.length = model.config.n_ctx // 2
    elif args.length > model.config.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % model.config.n_ctx)
    
    # import pdb;pdb.set_trace()

    # f = open('../result/generated_sample5.txt','w')
    hyp = []
    ref = []

    f = open('../result/'+args.output_dir+'.txt','w')
    f_ref = open('../result/reference.txt','w')
    counter=0
    for x,type_x,pos_x,lm_x,input_len in test_loader:
        if counter>10:
            break
        counter+=1
        print(counter)
        # f.write('='*5+'INPUT'+str(counter)+'='*5+'\n')
        # f.write(tokenizer.decode(x[0].tolist()[:input_len]))
        # f.write('\n')
        print('='*5+'INPUT'+str(counter)+'='*5+'\n')
        print(tokenizer.decode(x[0].tolist()[:input_len]))
        print('='*5+'GROUND'+str(counter)+'='*5+'\n')
        print(tokenizer.decode(x[0].tolist()[input_len:]))
        context_tokens = x[0][:input_len+1] # at evaluation stage, the input is without the ground truth
        generated = 0
        for i in range(args.nsamples // args.batch_size):

            out = sample_sequence(
                # model=model, length=args.length,
                model=model,length=int(0.5*len(context_tokens)),
                context=context_tokens,
                start_token=None,
                batch_size=args.batch_size,
                temperature=args.temperature, top_k=args.top_k, device=device
            )
            
            out = out[:, len(context_tokens):].tolist() # the generated result

            ref.append(tokenizer.decode(x[0].tolist()[len(context_tokens):-1]))
            f_ref.write(tokenizer.decode(x[0].tolist()[len(context_tokens):-1]))
            f_ref.write('\n')

            # f.write('='*5+'OUTPUT'+str(counter)+'='*5+'\n')
            hyp.append(tokenizer.decode(out[0]))
            f.write(tokenizer.decode(out[0]))
            f.write('\n')
            print('='*5+'OUTPUT'+str(counter)+'='*5+'\n')
            print(tokenizer.decode(out[0]))
    f.close()
    f_ref.close()
            # for i in range(args.batch_size):
            #     generated += 1
            #     text = tokenizer.decode(out[i])
            #     print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
            #     print(text)
    with open('result.txt','a') as f_result:
        rouge = Rouge()
        scores = rouge.get_scores(hyp, ref,avg=True)
        print("ROUGE",scores)
        f_result.write(args.model_dir+'\n')
        f_result.write(str(scores))
        f_result.write('\n')
            
    # while True:
    #     context_tokens = []
    #     if not args.unconditional:
    #         raw_text = input("Model prompt >>> ")
    #         while not raw_text:
    #             print('Prompt should not be empty!')
    #             raw_text = input("Model prompt >>> ")
    #         context_tokens = tokenizer.encode(raw_text)
    #         generated = 0
    #         for _ in range(args.nsamples // args.batch_size):
    #             out = sample_sequence(
    #                 model=model, length=args.length,
    #                 context=context_tokens,
    #                 start_token=None,
    #                 batch_size=args.batch_size,
    #                 temperature=args.temperature, top_k=args.top_k, device=device
    #             )
    #             out = out[:, len(context_tokens):].tolist()
    #             for i in range(args.batch_size):
    #                 generated += 1
    #                 text = tokenizer.decode(out[i])
    #                 print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
    #                 print(text)
    #         print("=" * 80)

    #     else:
    #         generated = 0
    #         for _ in range(args.nsamples // args.batch_size):
    #             out = sample_sequence(
    #                 model=model, length=args.length,
    #                 context=None,
    #                 start_token=tokenizer.encoder['<|endoftext|>'],
    #                 batch_size=args.batch_size,
    #                 temperature=args.temperature, top_k=args.top_k, device=device
    #             )
    #             out = out[:,1:].tolist()
    #             for i in range(args.batch_size):
    #                 generated += 1
    #                 text = tokenizer.decode(out[i])
    #                 print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
    #                 print(text)
    #         print("=" * 80)

if __name__ == '__main__':
    run_model()



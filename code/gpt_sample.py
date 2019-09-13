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
from pytorch_transformers import GPT2LMHeadModel, GPT2Tokenizer
from rouge import Rouge 
from utils import clean_text,text_standardize,values_lexicon_encode
from gpt_loader import GptDataset,collate_fn,GptDataset_aug, GptDataset_keyword, collate_fn_keyword

USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

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

def get_topic_keywords(meta):
    # TODO: temperary function
    keywords_up = []
    keywords_down = []
    if meta[1]=='Weight management':
        keywords_up += [6551, 4483, 2057, 9799, 4425, 4461, 4255, 5517]
        keywords_down += [46040, 21856, 2526, 13230, 7523, 15220]
    if meta[1]=='Smoking cessation':
        keywords_up += [46040, 21856, 2526, 13230, 7523, 15220]
        keywords_down += [6551, 4483, 2057, 9799, 4425, 4461, 4255, 5517]
    return keywords_up, keywords_down

def sample_sequence(model, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, modified_decoding=False,value_word_relation=None,device='cuda', sample=True,meta=None, key_word=None):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
        context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)

    keywords_up, keywords_down = get_topic_keywords(meta)
    prev = context
    output = context
    past = None

    # === get topic score ===
    if value_word_relation != None:
        value2word,word2value = value_word_relation
        topic_scores = {}
        topic_words = []
        for t in context[0]:
            token = t.tolist()
            if token in word2value:
                topic_words.append(token)
                if word2value[token] not in topic_scores:
                    topic_scores[word2value[token]]=1
                else:
                    topic_scores[word2value[token]]+=1
    # ========

    with torch.no_grad():
        for i in trange(length):
            logits, past = model(prev, past=past, key_word=key_word)
            logits = logits[:, -1, :] / temperature # torch.Size([1, 50257])
            logits = top_k_logits(logits, k=top_k) 
            log_probs = F.softmax(logits, dim=-1) # torch.Size([1, 50257])
            #=== modify probability after softmax ==========
            if modified_decoding:
                # logits[0][keywords_down]*=1.2 # multiply, lower logit, lower prob
                # logits[0][keywords_down]-= 100 # eliminate undesired word 
                # logits[0][keywords_up]/=1.2
                # import pdb;pdb.set_trace()
                
                if value_word_relation!=None:
                    for topic,score in topic_scores.items():
                        log_probs[0][value2word[topic]]*=1+(0.1*score)
                    log_probs[0][topic_words] *= 2
                        
            #===================
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1) # no need to normalize
                # import pdb;pdb.set_trace()
                prev_token = prev[0][0].tolist()
                if prev_token in word2value:    
                    print(prev_token)
                    topic = word2value[prev_token]
                    topic_words_i = [x for x in topic_words if x in value2word[topic]]
                    log_probs[0][topic_words_i] *= 2
                    prev = torch.multinomial(log_probs, num_samples=1) # no need to normalize

            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)
            output = torch.cat((output, prev), dim=1)
            if prev[0][0] in [50256]:
                break
    return output

def run_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='345M_Alex', help='pretrained model name or path to local checkpoint')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument('--unconditional', action='store_true', help='If true, unconditional generation.')
    parser.add_argument('--output_dir',type=str,default='generate', help="The name of the output file.")
    parser.add_argument('--modified_decoding', action='store_true')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--special_input',type=str,default='x_y_meta')
    parser.add_argument('--keyword', action='store_true')
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
    model_dir = '/data/chuancen/LIT/models/'+args.model_dir
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    if USE_CUDA:
        model.cuda()
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    
    # ========== Prepare lexicon =============
    value2word, word2value = values_lexicon_encode(path='../data_processed/values_lexicon/values_lexicon.txt',tokenizer=tokenizer)
    # =============== Load & process data ==============
    if args.augment:
        print("Using augmented data.")
        pickle_handler = open('/data/chuancen/LIT/mi_counselling/data_processed/x_y_meta_aug','rb')
        x_y_meta = pickle.load(pickle_handler)
        gpt_data = GptDataset_aug(x_y_meta,tokenizer) # use the name of output, it is depend on how is the trained model
    elif args.keyword:
        print("Using keyword cross attention")
        pickle_handler = open('/data/chuancen/LIT/mi_counselling/data_processed/x_y_meta_keyword', 'rb')
        # pickle_handler = open('/Users/shensq/Google Drive/Research/mi_counselling/data_processed/x_y_meta_keyword', 'rb')
        x_y_meta = pickle.load(pickle_handler)
        gpt_data = GptDataset_keyword(x_y_meta, tokenizer)
    else:
        pickle_handler = open('/data/chuancen/LIT/mi_counselling/data_processed/'+args.special_input,'rb')
        x_y_meta = pickle.load(pickle_handler)
        gpt_data = GptDataset(x_y_meta,tokenizer,args.output_dir) # use the output model name as pattern name

    print("Dataset initialized.")
    test_size  = int(len(gpt_data)*0.10)
    val_size = int(len(gpt_data)*0.05)
    gpt_train,gpt_test,gpt_val = torch.utils.data.random_split(gpt_data,[len(gpt_data)-test_size-val_size,test_size,val_size])

    # data_loader = DataLoader(dataset=gpt_train,batch_size=args.train_batch_size,shuffle=True,drop_last=True,collate_fn=collate_fn)
    if args.keyword:
        test_loader = DataLoader(dataset=gpt_test, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                 collate_fn=collate_fn_keyword)
    else:
        test_loader = DataLoader(dataset=gpt_test,batch_size=args.batch_size,shuffle=True,drop_last=True,collate_fn=collate_fn)
# ====

    model.eval()

    if args.length == -1:
        args.length = model.config.n_ctx // 2
    elif args.length > model.config.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % model.config.n_ctx)

    # f = open('../result/generated_sample5.txt','w')
    hyp = []
    ref = []

    f = open('../result/'+args.output_dir+'.txt','w')
    f_ref = open('../result/reference_'+args.output_dir+'.txt','w')
    counter=0
    for sample in test_loader:
        if args.keyword:
            x, type_x, pos_x, lm_x, x_len, meta, keyword_x = sample
        else:
            x, type_x, pos_x, lm_x, x_len, meta = sample
            keyword_x = x

        input_len = x_len[0]
        if counter>=1000:
            break
        counter+=1
        print(counter)
        # f.write('='*5+'INPUT'+str(counter)+'='*5+'\n')
        # f.write(tokenizer.decode(x[0].tolist()[:input_len]))
        # f.write('\n')

        # print('='*5+'INPUT'+str(counter)+'='*5+'\n')
        # print(tokenizer.decode(x[0].tolist()[:input_len]))
        # print('='*5+'GROUND'+str(counter)+'='*5+'\n')
        # print(tokenizer.decode(x[0].tolist()[input_len:]))
        context_tokens = x[0][:input_len+1] # at evaluation stage, the input is without the ground truth
        generated = 0
        for i in range(args.nsamples // args.batch_size):
            decode_length = int(len(context_tokens))
            # if args.augment:
            #     decode_length = int(0.5 * (5/6) * len(context_tokens))
            out = sample_sequence(
                # model=model, length=args.length,
                model=model,length=decode_length,
                context=context_tokens,
                start_token=None,
                batch_size=args.batch_size,
                temperature=args.temperature, top_k=args.top_k, modified_decoding=args.modified_decoding,
                value_word_relation=(value2word,word2value),device=device,meta=meta[0][0], key_word=keyword_x # an extra index for *meta
            )
            
            out = out[:, len(context_tokens):-1].tolist() # the generated result,get rid of eos

            ref.append(tokenizer.decode(x[0].tolist()[len(context_tokens):-1]))
            f_ref.write(tokenizer.decode(x[0].tolist()[len(context_tokens):-1]))
            f_ref.write('\n')

            # f.write('='*5+'OUTPUT'+str(counter)+'='*5+'\n')
            hyp.append(tokenizer.decode(out[0]))
            f.write(tokenizer.decode(out[0]))
            f.write('\n')
            # print('='*5+'OUTPUT'+str(counter)+'='*5+'\n')
            # print(tokenizer.decode(out[0]))
    f.close()
    f_ref.close()
            # for i in range(args.batch_size):
            #     generated += 1
            #     text = tokenizer.decode(out[i])
            #     print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
            #     print(text)
    
    # ======= Recording the experiment results
    with open('../result/rouge.txt','a') as f_result:
        rouge = Rouge()
        print(len(hyp))
        print(len(ref))
        scores = rouge.get_scores(hyp, ref,avg=True)
        print("ROUGE",scores)
        import time 
        f_result.write(time.asctime()+'\n')
        f_result.write(args.model_dir+'\n')
        f_result.write(str(scores))
        f_result.write('\n')
    # ================
    import sys
    sys.path.append('/data/chuancen/pip_package')
    import nltk
    from nltk.translate.meteor_score import meteor_score
    nltk.data.path.append('/data/chuancen/pip_package/nltk_data')
    print("#ref{} #hyp{}".format(len(ref),len(hyp)))
    meteor_sum = 0
    for i in range(min(len(ref),len(hyp))):
        meteor_sum += meteor_score([ref[i]],hyp[i])

    meteor_sum/=min(len(ref),len(hyp))
    print(meteor_sum)   

if __name__ == '__main__':
    run_model()



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
from utils import clean_text,text_standardize
from gpt_loader import GptDataset,collate_fn

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

def sample_sequence(model, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, device='cuda', sample=True,meta=None):
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
    with torch.no_grad():
        for i in trange(length):
            logits, past = model(prev, past=past)
            logits = logits[:, -1, :] / temperature # torch.Size([1, 50257])
            #===
            # logits[0][keywords_down]*=1.2 # multiply, lower logit, lower prob
            logits[0][keywords_down]-= 100 # eliminate undesired word 
            # logits[0][keywords_up]/=1.2
            #===
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
    parser.add_argument('--model_dir', type=str, default='345M_Alex', help='pretrained model name or path to local checkpoint')
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
    model_dir = '/data/chuancen/LIT/models/'+args.model_dir
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    if USE_CUDA:
        model.cuda()
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)

    # =============== Load & process data ==============
    pickle_handler = open('/data/chuancen/LIT/mi_counselling/data_processed/x_y_meta','rb')
    x_y_meta = pickle.load(pickle_handler)
    gpt_data = GptDataset(x_y_meta,tokenizer,args.output_dir) # use the name of output, it is depend on how is the trained model
    print("Dataset initialized.")
    test_size  = int(len(gpt_data)*0.10)
    val_size = int(len(gpt_data)*0.05)
    gpt_train,gpt_test,gpt_val = torch.utils.data.random_split(gpt_data,[len(gpt_data)-test_size-val_size,test_size,val_size])

    # data_loader = DataLoader(dataset=gpt_train,batch_size=args.train_batch_size,shuffle=True,drop_last=True,collate_fn=collate_fn)
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
    for x,type_x,pos_x,lm_x,input_len,*meta in test_loader:
        input_len = input_len[0]
        if counter>1000:
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

            out = sample_sequence(
                # model=model, length=args.length,
                model=model,length=int(0.5*len(context_tokens)),
                context=context_tokens,
                start_token=None,
                batch_size=args.batch_size,
                temperature=args.temperature, top_k=args.top_k, device=device,meta=meta[0][0] # an extra index for *meta
            )
            
            out = out[:, len(context_tokens):].tolist() # the generated result

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
    with open('../result/rouge.txt','a') as f_result:
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



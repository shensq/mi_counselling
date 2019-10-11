#!/usr/bin/env python3
import sys
sys.path.insert(0,'/home/shensq/LIT/pip_package') # make sure the modified version of pytorch_transformer
import pytorch_transformers
# assert pytorch_transformers.__file__[-36:]=='pip_package/transformers/__init__.py'
from pytorch_transformers import GPT2LMHeadModel, GPT2Tokenizer
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
from rouge import Rouge 
from utils import clean_text,text_standardize,values_lexicon_encode
from gpt_loader import GptDataset,collate_fn,GptDataset_aug, GptDataset_keyword, collate_fn_keyword
import nltk
from nltk.translate.meteor_score import meteor_score

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

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def sample_sequence(model, length, context, start_token=None, batch_size=1, modified_decoding=False,
                        value_word_relation=None, meta=None, key_word=None, num_samples=1, temperature=1,
                        top_k=0, top_p=0.0, device='cuda'):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    prev = context
    past = None

    with torch.no_grad():
        for i in trange(length):
            inputs = {'input_ids': generated, 'past': None, 'key_word': key_word}
            logits, past = model(**inputs)
            next_token_logits = logits[0, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            while (i == 0) and (next_token[0] == 50256):
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
            prev = next_token.unsqueeze(0)
            if next_token[0] in [50256]:
                break
    return generated

def sample_sequence_old(model, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, modified_decoding=False,value_word_relation=None,device='cuda', sample=True,meta=None, key_word=None):
    """ Generating a sequence until reaching the maximum length or getting an <eos> token.

    The generation is in an auto-regressive way. The initial input is the former utterances, w/o retrieved sentences or
    phrases.

    :param model: The PyTorch model to be used.
    :param length: The maximum decoding length
    :param start_token:
    :param batch_size:
    :param context: Tensor. The word tokens that the generation conditioned on.
    :param temperature: Double. The lower the temperature, the more likely the token with the highest probability is picked.
    :param top_k: Int. Sample from only the top-k probabilities to avoid unexpected result due to randomness
    :param modified_decoding: Boolean, whether is modify the prob based on lexicon.
    :param value_word_relation: Tuple. Contains word-'value lexicon' relationship.
    :param device: String. Specify the device to put the model for PyTorch
    :param sample: Boolean. To do sampling or not.
    :param meta: Tuple. Contains the meta information of the current input sample.
    :param key_word: A list of key phrases that will be used for cross-attention decoding.
    :return: TODO what is the output?
    """
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
            logits = logits[:, -1, :] / temperature # torch.Size([1, 50257]). next_token_logits
            logits = top_k_logits(logits, k=top_k)  # filtered_logits
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
                prev = torch.multinomial(log_probs, num_samples=1) # no need to normalize. sample the next token. (1,1)
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
            output = torch.cat((output, prev), dim=1) # con
            if prev[0][0] in [50256]:
                break
    return output

def load_model_data(args):
    #  === prepare data and model
    # ====== Load GPT2 model ========
    model_dir = '../models/'+args.model_dir
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    if USE_CUDA:
        model.cuda()
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    # ========== Prepare lexicon =============
    value2word, word2value = values_lexicon_encode(path='../data_processed/values_lexicon/values_lexicon.txt',tokenizer=tokenizer)
    # =============== Load & process data ==============
    if args.augment:
        print("Using augmented data.")
        pickle_handler = open('../data_processed/x_y_meta_aug','rb')
        x_y_meta = pickle.load(pickle_handler)
        gpt_data = GptDataset_aug(x_y_meta,tokenizer) # use the name of output, it is depend on how is the trained model
    elif args.keyword:
        print("Using keyword cross attention")
        pickle_handler = open('../data_processed/x_y_meta_keyword', 'rb')
        # pickle_handler = open('/Users/shensq/Google Drive/Research/mi_counselling/data_processed/x_y_meta_keyword', 'rb')
        x_y_meta = pickle.load(pickle_handler)
        gpt_data = GptDataset_keyword(x_y_meta, tokenizer)
    else:
        pickle_handler = open('../data_processed/'+args.special_input, 'rb')
        x_y_meta = pickle.load(pickle_handler)
        gpt_data = GptDataset(x_y_meta,tokenizer,args.output_dir, num_turns=args.num_turns) # use the output model name as pattern name

    print("Dataset initialized.")
    test_size  = int(len(gpt_data)*0.10)
    val_size = int(len(gpt_data)*0.05)
    gpt_train,gpt_test,gpt_val = torch.utils.data.random_split(gpt_data,[len(gpt_data)-test_size-val_size,test_size,val_size])
    if args.keyword:
        test_loader = DataLoader(dataset=gpt_test, batch_size=args.batch_size, shuffle=False, drop_last=True,
                                 collate_fn=collate_fn_keyword)
    else:
        test_loader = DataLoader(dataset=gpt_test,batch_size=args.batch_size,shuffle=False,drop_last=True,collate_fn=collate_fn)

    return model, tokenizer, test_loader

def run_model(args, model, tokenizer, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    if args.length == -1:
        args.length = model.config.n_ctx // 2
    elif args.length > model.config.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % model.config.n_ctx)

    hyp = []
    ref = []
    context = []
    f = open('../result/'+args.output_dir+'.txt','w')
    f_ref = open('../result/reference_'+args.output_dir+'.txt','w')
    
    for i,sample in enumerate(test_loader):
        if args.keyword:
            x, type_x, pos_x, lm_x, x_len, meta, keyword_x = sample
        else:
            x, type_x, pos_x, lm_x, x_len, meta = sample
            keyword_x = None
        input_len = x_len[0] # The number of tokens of the context utterances
        context_tokens = x[0][:input_len+1] # at evaluation stage, the input is without the ground truth
        generated = 0
        for i in range(args.nsamples // args.batch_size):
            decode_length = int(len(context_tokens))
            # if args.augment:
            #     decode_length = int(0.5 * (5/6) * len(context_tokens))
            out = sample_sequence(
                model=model,length=decode_length,
                context=context_tokens,
                start_token=None,
                batch_size=args.batch_size,
                temperature=args.temperature, top_k=args.top_k, modified_decoding=args.modified_decoding,
                value_word_relation=None,device=device,meta=meta[0][0], key_word=keyword_x # an extra index for *meta
            )           
            out = out[:, len(context_tokens):-1].tolist() # the generated result,get rid of eos

            ref.append(tokenizer.decode(x[0].tolist()[len(context_tokens):-1]))
            f_ref.write(tokenizer.decode(x[0].tolist()[len(context_tokens):-1]))
            f_ref.write('\n')

            hyp.append(tokenizer.decode(out[0]))
            f.write(tokenizer.decode(out[0]))
            f.write('\n')

            context.append(tokenizer.decode(x[0].tolist()[:len(context_tokens)]))
    f.close()
    f_ref.close()
    return hyp, ref, context

def calculate_metric(hyp, ref, context, effective_length=1024):
    # ===== Calculate rouge ========
    with open('../result/rouge.txt','a') as f_result:
        rouge = Rouge()
        print(len(hyp))
        print(len(ref))
        hyp, ref = zip(*[(x,y) for x,y in zip(hyp, ref) if len(x)>3 and len(y)>3])
        print(len(hyp))
        hyp = [x[:effective_length] for x in hyp]
        ref = [x[:effective_length] for x in ref]
        scores = rouge.get_scores(hyp, ref,avg=True)
        print("ROUGE",scores)
        import time 
        f_result.write(time.asctime()+'\n')
        f_result.write(args.model_dir+ '\t' + str(effective_length) +'\n')
        f_result.write(str(scores))
        f_result.write('\n')
    # ====== Calculate Meteor =========
    print("#ref{} #hyp{}".format(len(ref),len(hyp)))
    meteor_sum = 0
    for i in range(min(len(ref),len(hyp))):
        meteor_sum += meteor_score([ref[i]],hyp[i])

    meteor_sum/=min(len(ref),len(hyp))
    print(meteor_sum)   

def rouge_rank(hyp, ref, context):
    rouge = Rouge()
    # import pdb;pdb.set_trace()
    hyp, ref = zip(*[(x,y) for x,y in zip(hyp, ref) if len(x)>3 and len(y)>3])
    scores = rouge.get_scores(hyp, ref,avg=False) # type: list
    scores_content = zip(scores, hyp, ref, context, range(len(hyp)))
    scores_content = sorted(scores_content, key=lambda x:x[0]['rouge-1']['f'], reverse=True)
    return scores_content

if __name__ == '__main__':
    USE_CUDA = torch.cuda.is_available()
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    logger = logging.getLogger(__name__)

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='345M_Alex', help='pretrained model name or path to local checkpoint')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument('--output_dir',type=str,default='generate', help="The name of the output file.")
    parser.add_argument('--modified_decoding', action='store_true')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--special_input',type=str,default='x_y_meta_10turn')
    parser.add_argument('--keyword', action='store_true')
    parser.add_argument('--num_turns', type=int, default=5)
    args = parser.parse_args()
    if args.batch_size == -1:
        args.batch_size = 1
    assert args.nsamples % args.batch_size == 0
    print(args)

    # Setup the random seeds.
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    model, toknenizer, test_loader = load_model_data(args)

    hyp, ref, context = run_model(args, model, toknenizer, test_loader)
    sample_ranked = rouge_rank(hyp, ref, context)
    with open("../data_processed/rouge_rank_" + args.model_dir,'wb') as f:
        pickle.dump(sample_ranked, f)
    calculate_metric(hyp, ref, context)
    calculate_metric(hyp, ref, context, 5)




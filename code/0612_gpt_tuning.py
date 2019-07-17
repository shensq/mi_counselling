# Path to the pytorch checkpoint
# /Users/shensq/Documents/LIT_ai_counseling/gpt2/models/pytorch_345M'

import re
import argparse
import torch
import pickle
import os
from pytorch_pretrained_bert import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel,OpenAIAdam, WEIGHTS_NAME, CONFIG_NAME
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
from tqdm import tqdm, trange
import random 

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)
USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor

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
        
        
        x += [ref_start] + self.y_encoded[index] + [50256]
        type_x += [ref_start]*(len(self.y_encoded[index])+2)
        lm_x += [-1] + self.y_encoded[index] + [-1]
        
        position_x = list(range(len(x)))
        
        x = torch.Tensor(x)
        type_x = torch.Tensor(type_x)
        position_x = torch.Tensor(position_x)
        lm_x = torch.Tensor(lm_x)
        x_len = x.shape[0]
        
        return x,type_x,position_x,lm_x
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",default='345M_after_Alex',type=str,required=False,
                        help="The directory of the model to be tuned.")
    parser.add_argument("--output_dir", default='mi_tuned', type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=6.25e-5)
    parser.add_argument('--warmup_proportion', type=float, default=0.002)
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lm_coef', type=float, default=0.9)
    parser.add_argument('--n_valid', type=int, default=374)

    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()
    print(args)

    # ====== Set random seed =========
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # ====== Load GPT2 model ========
    model_dir = '../gpt2/models/'+args.model_dir
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    if USE_CUDA:
        model.cuda()
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

    data_loader = DataLoader(dataset=gpt_train,batch_size=args.train_batch_size,shuffle=True,drop_last=True,collate_fn=collate_fn)
    test_loader = DataLoader(dataset=gpt_test,batch_size=4,shuffle=True,drop_last=True,collate_fn=collate_fn)

    # ========== Prepare optimizer =============
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    num_train_optimization_steps = len(gpt_train) * args.num_train_epochs // args.train_batch_size
    optimizer = OpenAIAdam(optimizer_grouped_parameters,
                        lr=args.learning_rate,
                        warmup=args.warmup_proportion,
                        max_grad_norm=args.max_grad_norm,
                        weight_decay=args.weight_decay,
                        t_total=num_train_optimization_steps)

    # Training
    model.train()
    exp_average_loss = None
    
    counter=0
    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        # tqdm_bar = tqdm(data_loader, desc="Training")
        for x,type_x,pos_x,lm_x,x_len in tqdm(data_loader):
            if counter>0:
                break
            # counter+=1
            # print("Get data")
            loss = model(x, position_ids=pos_x, token_type_ids=type_x, lm_labels=lm_x)
            # print("Forward pass")
            loss.backward()
            optimizer.step()
            # print("loss BP")
            optimizer.zero_grad()
            exp_average_loss = loss.item() if exp_average_loss is None else 0.7*exp_average_loss+0.3*loss.item()
            # tqdm_bar.desc = "Training loss: {:.2e} lr: {:.2e}".format(exp_average_loss, optimizer.get_lr()[0])
            # print(exp_average_loss)

        # ==== Save the model ====
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_dir = '../gpt2/models/'
        output_model_file = os.path.join(output_dir+args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir+args.output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(output_dir+args.output_dir)


if __name__ == '__main__':
    main()

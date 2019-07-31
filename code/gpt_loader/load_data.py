import torch
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
import sys
sys.path.append("..")
from utils import text_standardize


USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor


# ==== Code for data loading =====
class GptDataset(Dataset):
    """Take a list of samples with form [[x,...],y,meta]
    """
    # need 3 special tokens
    # # as <ref start> 2
    # $ as <speaker1> 3
    # % as <speaker2> 4
    # '<|endoftext|>' as <eos> 50256
    def _split(self,x_y_meta):
        x_all = []
        y_all = []
        meta_all = []
        for x,y,meta in x_y_meta:
            meta_all.append(meta)
            x_all.append([self.tokenizer.encode(text_standardize(x_i)) for x_i in x])
            y_all.append(self.tokenizer.encode(text_standardize(y)))

        return x_all,y_all,meta_all
    
    def _filter(self,x_all,y_all,meta_all,filter_mode=None):
        allowed_pattern = ['SR_only','CR_only','Smoking_only','Diet_only']
        data = zip(x_all,y_all,meta_all)
        if filter_mode not in allowed_pattern:
            data_filt = data
        if filter_mode=='SR_only':
            data_filt = [x for x in data if x[2][2]=='SR']
        if filter_mode=='CR_only':
            data_filt = [x for x in data if x[2][2]=='CR']
        if filter_mode=='Smoking_only':
            data_filt = [x for x in data if x[2][1]=='Smoking cessation']
        if filter_mode=='Diet_only':
            data_filt = [x for x in data if x[2][1]=='Weight management']
        x_filt,y_filt,meta_filt = zip(*data_filt)
        return x_filt, y_filt, meta_filt

    def __init__(self,x_y_meta,tokenizer,filter_mode=None,num_turns=5):
        
        self.x_y_meta = x_y_meta
        self.num_turns = num_turns
        self.tokenizer = tokenizer
        self.x_encoded,self.y_encoded,self.meta = self._split(x_y_meta)
        self.x_encoded,self.y_encoded,self.meta = self._filter(self.x_encoded,self.y_encoded,self.meta,filter_mode)
        self.ref_start, self.speaker1,self.speaker2,self.eos = 2,3,4,50256

    def __getitem__(self,index):
        x = []
        type_x = []
        lm_x = []
        is_speaker1 = bool(len(self.x_encoded[index])%2) # which speaker start the conversation

        for utt in self.x_encoded[index]:
            if is_speaker1: # add the prefix special token for each utterance
                x+=[self.speaker1]
                type_x += [self.speaker1]*(len(utt)+1)
            else:
                x+=[self.speaker2]
                type_x += [self.speaker2]*(len(utt)+1)
            x += utt
            is_speaker1 = not is_speaker1
        lm_x += [-1]*len(x) # all position for the input is masked for loss calculation

        total_input_length = len(x)

        x += [self.ref_start] + self.y_encoded[index] + [self.eos]

        type_x += [self.ref_start]*(len(self.y_encoded[index])+2)
        lm_x += [-1] + self.y_encoded[index] + [self.eos]
        position_x = list(range(len(x)))

        x = torch.Tensor(x)
        type_x = torch.Tensor(type_x)
        position_x = torch.Tensor(position_x)
        lm_x = torch.Tensor(lm_x)
        x_len = x.shape[0]
        
        return x,type_x,position_x,lm_x,total_input_length,self.meta[index]

    def __len__(self):
        return len(self.x_encoded)

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
    src_seqs, trg_seqs, pos_seqs,lm_seqs,total_input_length,meta = zip(*data)

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
    return Variable(LongTensor(src_seqs)), Variable(LongTensor(trg_seqs)), Variable(LongTensor(pos_seqs)),Variable(LongTensor(lm_seqs)), total_input_length, meta

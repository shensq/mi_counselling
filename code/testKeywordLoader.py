import pickle
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel
import sys
# sys.path.append("/Users/shensq/Google Drive/Research/mi_counselling/code")
from gpt_loader import GptDataset_keyword, collate_fn_keyword, GptDataset, collate_fn
import torch
import random
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader

random.seed(42)
torch.random.manual_seed(42)
torch.cuda.manual_seed(42)
# ======= Prepare ==========

USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor


pickle_handler = open('../data_processed/x_y_meta_keyword', 'rb')
# pickle_handler = open('../data_processed/x_y_meta', 'rb')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
x_y_meta = pickle.load(pickle_handler)

gpt_data = GptDataset_keyword(x_y_meta, tokenizer)
# gpt_data = GptDataset(x_y_meta, tokenizer)

print("Dataset initialized. There are {} samples.".format(len(gpt_data)))

test_size = int(len(gpt_data) * 0.10)
val_size = int(len(gpt_data) * 0.05)
gpt_train, gpt_test, gpt_val = torch.utils.data.random_split(gpt_data, [len(gpt_data) - test_size - val_size, test_size,
                                                                        val_size])
model = GPT2LMHeadModel.from_pretrained('gpt2')

data_loader = DataLoader(dataset=gpt_train,batch_size=1,shuffle=True,drop_last=True,collate_fn=collate_fn_keyword)

counter = 0
for x, type_x, pos_x, lm_x, x_len, _ , key_word in data_loader:
    if counter > 10:
        break
    counter += 1
    # print(tokenizer.decode(keyword_x[0].tolist()))
    loss = model(x, position_ids=pos_x, token_type_ids=type_x, labels=lm_x, key_word=key_word)[0]
    print(loss)
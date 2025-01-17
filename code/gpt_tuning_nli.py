# Path to the pytorch checkpoint
# /Users/shensq/Documents/LIT_ai_counseling/gpt2/models/pytorch_345M'
import sys
sys.path.insert(0,'/home/shensq/LIT/pip_package')
import re
import argparse
import torch
import pickle
import os
import pytorch_transformers
from pytorch_transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel,AdamW, WEIGHTS_NAME, CONFIG_NAME
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
from tqdm import tqdm, trange
import random 
from utils import clean_text,text_standardize
from gpt_loader import GptDataset,collate_fn,collate_fn_nli,GptDataset_nli,SnliDataset
from model import GPT2ClassHeadsModel
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging

def eval(data_loader, model):
    # tqdm_bar = tqdm(data_loader, desc="Evaluating")
    accuracy = 0
    total = 0
    exp_average_loss = None
    model.eval()

    with torch.no_grad():
        for x, type_x, pos_x, lm_x, label in tqdm(data_loader):
            loss, logits = model(x, position_ids=pos_x, token_type_ids=type_x, labels=label)

            total += 1
            if torch.argmax(logits, dim=1).item() == label.item():
                accuracy += 1

            exp_average_loss = loss.item() if exp_average_loss is None else 0.7 * exp_average_loss + 0.3 * loss.item()
            # tqdm_bar.update(1)
            # tqdm_bar.set_postfix(loss=exp_average_loss, correct=accuracy)

    accuracy /= total
    model.train()
    return accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",default='345M_origin', type=str,required=False,
                        help="The directory of the model to be tuned.")
    parser.add_argument("--output_dir", default='mi_tuned', type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    # parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--learning_rate', type=float, default=6.25e-5)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--snli', action='store_true')
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()
    print(args)

    # ====== Set random seed =========
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    #======= Prepare ==========
    logging.basicConfig(level=logging.INFO)
    USE_CUDA = torch.cuda.is_available()
    FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
    ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor

    
    # ====== Load GPT2 model ========
    model_dir = "../models/" + args.model_dir
    model = GPT2ClassHeadsModel.from_pretrained(model_dir)
    # model = GPT2ClassHeadsModel.from_pretrained('gpt2')
    if USE_CUDA:
        model.cuda()
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    print('Model loaded.')
    # =============== Load & process data ==============
    pickle_handler = open('../data_processed/x_y_meta_10turn', 'rb')
    x_y_meta = pickle.load(pickle_handler)
    if args.snli:
        print("Using SNLI data.")
        gpt_data = SnliDataset(tokenizer)  # use the output model name as pattern name
    else:
        print("Using mi data.")
        gpt_data = GptDataset_nli(x_y_meta, tokenizer, augment=True, num_turns=10)

    print("Dataset initialized.")
    print("samples:", len(gpt_data))
    test_size  = int(len(gpt_data)*0.10)
    val_size = int(len(gpt_data)*0.05)
    gpt_train,gpt_test,gpt_val = torch.utils.data.random_split(gpt_data,[len(gpt_data)-test_size-val_size,test_size,val_size])
    
    data_loader = DataLoader(dataset=gpt_train,batch_size=args.train_batch_size,shuffle=True,drop_last=True,collate_fn=collate_fn_nli)
    test_loader = DataLoader(dataset=gpt_test, batch_size=1, shuffle=False, drop_last=False, collate_fn=collate_fn_nli)
    val_loader = DataLoader(dataset=gpt_val, batch_size=1, shuffle=False, drop_last=False, collate_fn=collate_fn_nli)
    if args.eval:
        print(eval(test_loader, model))
        return

    # ========== Prepare optimizer =============
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and 'classifier' not in n], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and 'classifier' not in n], 'weight_decay': 0.0}
        ]

    optimizer_grouped_parameters_classifier = [
        {'params': [p for n, p in param_optimizer if 'classifier' in n and 'bias' not in n], 'weight_decay': 0.01, 'lr':args.learning_rate},
        {'params': [p for n, p in param_optimizer if 'classifier' in n and 'bias' in n], 'weight_decay': 0.00,'lr': args.learning_rate}
    ]
    num_train_optimization_steps = len(gpt_train) * args.num_train_epochs // args.train_batch_size
    num_warmup_steps = int(num_train_optimization_steps) * 0.1

    optimizer = AdamW(optimizer_grouped_parameters,lr=args.learning_rate,correct_bias=True)
    optimizer_classifier = AdamW(optimizer_grouped_parameters_classifier,lr=args.learning_rate,correct_bias=True)
    # scheduler = pytorch_transformers.optimization.WarmupCosineSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=num_train_optimization_steps,cycles=1.5)
    scheduler = pytorch_transformers.optimization.WarmupLinearSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=num_train_optimization_steps)

    # Training
    print("Start training.")
    model.train()
    exp_average_loss = None
    max_eval_accuracy = 0 
    early_terminate_counter = 0
    train_losses = []
    eval_losses = []
    for epo in trange(int(args.num_train_epochs), desc="Epoch"):
    # for epo in range(int(args.num_train_epochs)):
        tqdm_bar = tqdm(data_loader, desc="Training")
        accuracy = 0 
        for x,type_x,pos_x,lm_x,label in data_loader:
            # import pdb;pdb.set_trace()
            # for i in range(x.shape[0]):
            #     if label[i].item()==0:
            #         x[i].fill_(0)
            #     else:
            #         x[i].fill_(1)
            loss,logits = model(x, position_ids=pos_x, token_type_ids=type_x, labels=label)
            pred = torch.argmax(logits, dim=1)
            for i in range(x.shape[0]):
                if pred[i].item() == label[i].item():
                    accuracy += 1
            loss.backward()
            optimizer.step()
            optimizer_classifier.step()
            scheduler.step()
            optimizer.zero_grad()
            optimizer_classifier.zero_grad()
            exp_average_loss = loss.item() if exp_average_loss is None else 0.7*exp_average_loss+0.3*loss.item()
            tqdm_bar.update(1)
            tqdm_bar.set_postfix(loss=exp_average_loss,correct=accuracy)

        accuracy/=len(gpt_train)
        print("Accuracy for epoch {} is {}.\t Average loss:{}".format(epo,accuracy,exp_average_loss))
        train_losses.append([accuracy, exp_average_loss])

        eval_accuracy = eval(val_loader, model)
        print("Eval accuracy: {}".format(eval_accuracy))
        eval_losses.append(eval_accuracy)

        # if eval_accuracy < max_eval_accuracy:
        if False:
            print("eval accuracy decreasing!")
            early_terminate_counter += 1
            if early_terminate_counter > 10:
                break
        else:
            early_terminate_counter = 0
            max_eval_accuracy = eval_accuracy
            # ==== Save the model ====
            # Save a trained model, configuration and tokenizer
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

            # If we save using the predefined names, we can load using `from_pretrained`
            output_dir = '../models/'
            output_model_file = os.path.join(output_dir+args.output_dir, WEIGHTS_NAME)
            output_config_file = os.path.join(output_dir+args.output_dir, CONFIG_NAME)

            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)
            tokenizer.save_vocabulary(output_dir+args.output_dir)
    with open(output_dir+'exp_info.txt', 'wb') as f:
        d = {'args':args,'train_info':train_losses,'eval_info':eval_accuracy}
        pickle.dump(d, f)
                


if __name__ == '__main__':
    main()

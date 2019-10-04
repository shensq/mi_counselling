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
from utils import clean_text, text_standardize, construct_grouped_parameters, get_unfeezing_funcs
from gpt_loader import GptDataset,collate_fn,GptDataset_aug, GptDataset_keyword, collate_fn_keyword

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",default='345M_Alex',type=str,required=False,
                        help="The directory of the model to be tuned.")
    parser.add_argument("--output_dir", default='mi_tuned', type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=6.25e-5)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lm_coef', type=float, default=0.9)
    parser.add_argument('--n_valid', type=int, default=374)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--keyword', action='store_true')
    parser.add_argument('--special_input', type=str, default='x_y_meta')
    parser.add_argument('--first_K_tokens', type=int, default=1024)
    parser.add_argument('--use_disc_lr', action='store_true')
    parser.add_argument('--num_turns', type=int, default=5)
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
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
    model_dir = '../models/'+args.model_dir
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    # model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    if USE_CUDA:
        model.cuda()
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    print('Model loaded.')
    # =============== Load & process data ==============
    # # folder = '/home/shensq/gpt_tuning/multiturns_data/'
    # folder = '../data_processed/'
    # x_flat = pickle.load(open(folder+'x_flat','rb'))
    # y_all_join = pickle.load(open(folder+'y_all_join','rb'))

    if args.augment:
        print("Using augmented data")
        pickle_handler = open('../data_processed/x_y_meta_aug','rb')
        x_y_meta = pickle.load(pickle_handler)
        gpt_data = GptDataset_aug(x_y_meta,tokenizer)
    elif args.keyword:
        print("Using keyword cross attention")
        pickle_handler = open('../data_processed/x_y_meta_keyword','rb')
        x_y_meta = pickle.load(pickle_handler)
        gpt_data = GptDataset_keyword(x_y_meta, tokenizer)
    else:
        if args.special_input != 'x_y_meta':
            print("Using mutated data.")
            pickle_handler = open('../data_processed/'+args.special_input, 'rb')
        else:
            pickle_handler = open('../data_processed/x_y_meta_10turn', 'rb')
        x_y_meta = pickle.load(pickle_handler)
        gpt_data = GptDataset(x_y_meta,tokenizer, args.output_dir, num_turns=args.num_turns) # use the output model name as pattern name
    print("Dataset initialized. There are {} samples.".format(len(gpt_data)))

    test_size = int(len(gpt_data)*0.10)
    val_size = int(len(gpt_data)*0.05)
    gpt_train, gpt_test, gpt_val = torch.utils.data.random_split(gpt_data, [len(gpt_data)-test_size-val_size, test_size, val_size])

    if args.keyword:
        data_loader = DataLoader(dataset=gpt_train, batch_size=args.train_batch_size, shuffle=True, drop_last=True,
                                 collate_fn=collate_fn_keyword)
    else:
        data_loader = DataLoader(dataset=gpt_train, batch_size=args.train_batch_size, shuffle=True, drop_last=True,
                                 collate_fn=collate_fn)
        test_loader = DataLoader(dataset=gpt_test, batch_size=4, shuffle=True, drop_last=True, collate_fn=collate_fn)

    # ========== Prepare optimizer =============

    param_optimizer = list(model.named_parameters()) + list(model.lm_head.named_parameters()) # the gpt2 model from library has unnamed LM head.
    optimizer_grouped_parameters = construct_grouped_parameters(param_optimizer, args.learning_rate, use_discr=args.use_disc_lr)

    num_train_optimization_steps = len(gpt_train) * args.num_train_epochs // args.train_batch_size



    lm_funcs = get_unfeezing_funcs(optimizer_grouped_parameters, warmup_portion=args.warmup_proportion, total_steps = num_train_optimization_steps)

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lm_funcs)
    # num_warmup_steps = int(num_train_optimization_steps * 0.1)
    # scheduler = pytorch_transformers.optimization.WarmupLinearSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=num_train_optimization_steps)


    # Training
    print("Start training.")
    model.train()
    exp_average_loss = None
    progress_bar = trange(int(args.num_train_epochs), desc="Epoch", leave=True)
    for _ in progress_bar:
        for sample in tqdm(data_loader):
            if args.keyword:
                x, type_x, pos_x, lm_x, x_len, _, keyword_x = sample
            else:
                x, type_x, pos_x, lm_x, x_len, _ = sample
                keyword_x = x
            input_len = x_len[0]
            if x_len[0] > 1023:
                continue
            
            lm_x[:, x_len[0]+1+args.first_K_tokens:-1] = -1
            loss = model(x, position_ids=pos_x, token_type_ids=type_x, labels=lm_x, key_word=keyword_x)[0]
            loss.backward()
            optimizer.step()
            scheduler.step()
            # print("loss BP")
            optimizer.zero_grad()
            exp_average_loss = loss.item() if exp_average_loss is None else 0.7*exp_average_loss+0.3*loss.item()
            progress_bar.set_description("Training loss: {}".format(exp_average_loss))
            # exp_average_loss = loss.mean().item() if exp_average_loss is None else 0.7*exp_average_loss+0.3*loss.mean().item()
            # tqdm_bar.desc = "Training loss: {:.2e} lr: {:.2e}".format(exp_average_loss, optimizer.get_lr()[0])
            # print(exp_average_loss)

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


if __name__ == '__main__':
    main()

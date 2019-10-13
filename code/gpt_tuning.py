# Path to the pytorch checkpoint
# /Users/shensq/Documents/LIT_ai_counseling/gpt2/models/pytorch_345M'
import sys

sys.path.insert(0, '/home/shensq/LIT/pip_package')

import re
import argparse
import torch
import pickle
import os
import pytorch_transformers
from pytorch_transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, AdamW, WEIGHTS_NAME, CONFIG_NAME
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from tqdm import tqdm, trange
import random
from utils import clean_text, text_standardize, construct_grouped_parameters, get_unfreezing_funcs
from gpt_loader import GptDataset, collate_fn, GptDataset_aug, GptDataset_keyword, collate_fn_keyword

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging


def get_data(args, tokenizer, split_size):
    """
    Return the data loaders needed for training and evaluation.
    :param args: command line arguments.
    :param tokenizer: the tokenizer used in preparing the data.
    :param split_size: the portion of train, test, validation set.
    :return data_loader: The data loader for the training set.
    :return val_loader: The data loader for the validation set.
    """
    if args.augment:
        print("Using augmented data")
        pickle_handler = open('../data_processed/x_y_meta_aug', 'rb')
        x_y_meta = pickle.load(pickle_handler)
        gpt_data = GptDataset_aug(x_y_meta, tokenizer)
    elif args.keyword:
        print("Using keyword cross attention")
        pickle_handler = open('../data_processed/x_y_meta_keyword', 'rb')
        x_y_meta = pickle.load(pickle_handler)
        gpt_data = GptDataset_keyword(x_y_meta, tokenizer)
    elif args.special_input != 'x_y_meta':
        print("Using mutated data.")
        pickle_handler = open('../data_processed/' + args.special_input, 'rb')
        x_y_meta = pickle.load(pickle_handler)
        gpt_data = GptDataset(x_y_meta, tokenizer, args.output_dir, num_turns=args.num_turns)
    else:
        print("Using vanilla data.")
        pickle_handler = open('../data_processed/x_y_meta', 'rb')
        x_y_meta = pickle.load(pickle_handler)
        gpt_data = GptDataset(x_y_meta, tokenizer, args.output_dir, num_turns=args.num_turns)

    print("Dataset initialized. There are {} samples.".format(len(gpt_data)))
    test_size = int(len(gpt_data) * split_size['test'])
    val_size = int(len(gpt_data) * split_size['val'])
    gpt_train, gpt_test, gpt_val = torch.utils.data.random_split(gpt_data,
                                                                 [len(gpt_data) - test_size - val_size, test_size,
                                                                  val_size])
    if args.keyword:
        data_loader = DataLoader(dataset=gpt_train, batch_size=args.train_batch_size, shuffle=True, drop_last=True,
                                 collate_fn=collate_fn_keyword)
        val_loader = DataLoader(dataset=gpt_val, batch_size=1, shuffle=False, drop_last=False,
                                collate_fn=collate_fn_keyword)
    else:
        data_loader = DataLoader(dataset=gpt_train, batch_size=args.train_batch_size, shuffle=True, drop_last=True,
                                 collate_fn=collate_fn)
        val_loader = DataLoader(dataset=gpt_val, batch_size=1, shuffle=False, drop_last=False, collate_fn=collate_fn)
    return data_loader, val_loader


def evaluate(model, data_loader, use_keyword=None):
    """
    Evaluate the model on validation set.
    :param model: The model being training.
    :param data_loader: the data loader for validation set.
    :param use_keyword: whether the input contains keyword or not.
    :return: eval_loss: the average loss on the validation set.
    """
    model.eval()
    eval_loss = 0
    for sample in tqdm(data_loader):
        if use_keyword:
            x, type_x, pos_x, lm_x, x_len, _, keyword_x = sample
        else:
            x, type_x, pos_x, lm_x, x_len, _ = sample
            keyword_x = None
        loss = model(x, position_ids=pos_x, token_type_ids=type_x, labels=lm_x, key_word=keyword_x,
                     use_keyword=use_keyword)[0]
        eval_loss += loss.item()
    eval_loss /= len(data_loader)
    model.train()
    return eval_loss


def parse_arguments():
    """
    Parse command line argument using argparse.
    :return args: A parser object with hyper-parameters' name and their values.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default='345M_Alex', type=str, required=False,
                        help="The directory of the model to be tuned.")
    parser.add_argument("--output_dir", default='mi_tuned', type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_train_epochs', type=int, default=1)
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
    parser.add_argument('--use_unfreezing', action='store_true')
    parser.add_argument('--num_turns', type=int, default=5)
    args = parser.parse_args()
    print(args)
    return args

def load_model(args):
    """
    Load model and the corresponding tokenizer from pre-trained weight.
    :param args: The command line arguments.
    :return model: The main model.
    :return tokenzier: The tokenzier comes with the main model.
    """
    USE_CUDA = torch.cuda.is_available()
    # ====== Load GPT2 model ========
    model_dir = '../models/' + args.model_dir
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    # model = GPT2LMHeadModel.from_pretrained('gpt2')
    if USE_CUDA:
        model.cuda()
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    print('Model loaded.')
    return model, tokenizer

def main():
    args = parse_arguments()

    # ====== Set random seed =========
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # ======= Prepare ==========
    logging.basicConfig(level=logging.INFO)
    USE_CUDA = torch.cuda.is_available()
    FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
    ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor

    model, tokenizer = load_model(args)
    # =============== Load & process data ==============
    split_size = {'train': 0.8, 'test': 0.1, 'val': 0.05}
    data_loader, val_loader = get_data(args, split_size=split_size, tokenizer=tokenizer)
    # ========== Prepare optimizer =============
    # the gpt2 model from library has unnamed LM head. LM head's weights are tied to input embedding
    num_train_optimization_steps = len(data_loader) * args.num_train_epochs // args.train_batch_size

    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = construct_grouped_parameters(param_optimizer, args.learning_rate,
                                                                use_discr=args.use_disc_lr)

    lm_funcs = get_unfreezing_funcs(optimizer_grouped_parameters, warmup_portion=args.warmup_proportion,
                                    total_steps=num_train_optimization_steps, use_unfreezing=args.use_unfreezing)

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lm_funcs)

    # Training
    print("Start training.")
    model.train()
    exp_average_loss = None
    progress_bar = trange(int(args.num_train_epochs), desc="Epoch", leave=True)
    min_eval_loss = 100  # large enough number
    early_terminate_counter = 0
    for _ in progress_bar:
    # for _ in range(int(args.num_train_epochs)):
        for sample in tqdm(data_loader):
        # for sample in data_loader:
            if args.keyword:
                x, type_x, pos_x, lm_x, x_len, _, keyword_x = sample
            else:
                x, type_x, pos_x, lm_x, x_len, _ = sample
                keyword_x = None
            input_len = x_len[0]
            lm_x[:, x_len[0] + 1 + args.first_K_tokens:-1] = -1
            loss = model(x, position_ids=pos_x, token_type_ids=type_x, labels=lm_x, key_word=keyword_x,
                         use_keyword=args.keyword)[0]
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            exp_average_loss = loss.item() if exp_average_loss is None else 0.7 * exp_average_loss + 0.3 * loss.item()
            progress_bar.set_description("Training loss: {}".format(exp_average_loss))

        eval_loss = evaluate(model, val_loader, use_keyword=args.keyword)
        print("Eval loss: {}".format(eval_loss))
        if eval_loss < min_eval_loss:  # save the model only when the loss is the smallest
            early_terminate_counter = 0
            min_eval_loss = eval_loss
            # ==== Save the model ====
            # Save a trained model, configuration and tokenizer
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            # If we save using the predefined names, we can load using `from_pretrained`
            output_dir = '../models/'
            output_model_file = os.path.join(output_dir + args.output_dir, WEIGHTS_NAME)
            output_config_file = os.path.join(output_dir + args.output_dir, CONFIG_NAME)

            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)
            tokenizer.save_vocabulary(output_dir + args.output_dir)
        else:
            print("eval loss increasing!")
            early_terminate_counter += 1
            if early_terminate_counter > 5:  # if the eval loss does not decrease for 5 epochs, terminate early.
                return

if __name__ == '__main__':
    main()

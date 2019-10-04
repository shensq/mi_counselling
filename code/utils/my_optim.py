# def construct_grouped_parameters_old(param_optimizer, lr):
#     # discriminative lr
#     optimizer_grouped_parameters = []
#     all_name = []
#     no_decay = ['bias', 'ln_']
#     need_decay = lambda n: not any(nd in n for nd in no_decay)
#     in_block = lambda name, i: "h.{}.".format(i) in name
#
#     num_blocks = 12
#     for i in range(num_blocks):
#         group = {}
#         group['params'] = [p for n, p in param_optimizer if need_decay(n) and in_block(n, i)]
#         group['weight_decay'] = 0.01
#         group['tag'] = "h.{}.".format(i)
#         group['lr'] = lr * (0.5 ** (11 - i))  # soft mask
#         optimizer_grouped_parameters.append(group)
#
#         group = {}
#         group['params'] = [p for n, p in param_optimizer if not need_decay(n) and in_block(n, i)]
#         group['weight_decay'] = 0
#         group['tag'] = "h.{}.".format(i)
#         group['lr'] = lr * (0.5 ** (11 - i))  # soft mask
#         optimizer_grouped_parameters.append(group)
#
#     group = {}
#     group['params'] = [p for n, p in param_optimizer if "ln_f" in n]
#     group['weight_decay'] = 0
#     group['lr'] = lr
#     group['tag'] = "ln_f"
#     optimizer_grouped_parameters.append(group)
#
#     group = {}
#     group['params'] = [p for n, p in param_optimizer if n == 'weight']
#     group['weight_decay'] = 0.01
#     group['lr'] = lr
#     group['tag'] = "lm_head"
#     optimizer_grouped_parameters.append(group)
#     return optimizer_grouped_parameters
#
# def unfreezing_parameters(optimizer_grouped_parameters, epoch):
#     if epoch > 12:
#         return
#     for group in optimizer_grouped_parameters:
#         if group['tag'] == "h.{}.".format(12 - epoch):
#             group['lr'] *= 1e20


def get_unfeezing_funcs(optimizer_grouped_parameters, warmup_portion, total_steps):
    def get_lr_lambda(block):
        NUM_STAGES = 30
        warmup_start = int((24 - block) * total_steps / NUM_STAGES)
        warmup_length = int((total_steps - warmup_start) * warmup_portion)

        def lr_lambda(step):
            if step < warmup_start:
                return 0
            elif step < warmup_length + warmup_start:
                return float(step - warmup_start) / float(max(1, warmup_length))
            else:
                return max(0.0, float(total_steps - step) / float(max(1.0, total_steps - warmup_start - warmup_length)))

        return lr_lambda

    lm_funcs = []
    for group in optimizer_grouped_parameters:
        tag = group['tag']
        if tag[:2] != 'h.':
            lm_funcs.append(get_lr_lambda(24))
        else:
            try:
                block = int(tag[2:-1])  # get the block number of a parameters group
                lm_funcs.append(get_lr_lambda(block))
            except:
                print("Exists invalid block numbers while creating unfreezing scheme")
    return lm_funcs

def construct_grouped_parameters(param_optimizer, learning_rate, use_discr=True):
    # discriminative lr
    optimizer_grouped_parameters = []
    all_name = []
    no_decay = ['bias', 'ln_']
    need_decay = lambda n: not any(nd in n for nd in no_decay)
    in_block = lambda name, i: "h.{}.".format(i) in name

    num_blocks = 24
    for i in range(num_blocks):
        tag = "h.{}.".format(i)
        if use_discr:
            lr = learning_rate * (0.8 ** (num_blocks - i))
        else:
            lr = learning_rate

        group = {'lr': lr, 'tag': tag, 'weight_decay': 0.01}
        group['params'] = [p for n, p in param_optimizer if need_decay(n) and in_block(n, i)]
        optimizer_grouped_parameters.append(group)

        group = {'lr': lr, 'tag': tag, 'weight_decay': 0}
        group['params'] = [p for n, p in param_optimizer if not need_decay(n) and in_block(n, i)]
        optimizer_grouped_parameters.append(group)

    group = {}
    group['params'] = [p for n, p in param_optimizer if "ln_f" in n]
    group['weight_decay'] = 0
    group['lr'] = learning_rate
    group['tag'] = "ln_f"
    optimizer_grouped_parameters.append(group)

    group = {}
    group['params'] = [p for n, p in param_optimizer if n == 'weight']
    group['weight_decay'] = 0.01
    group['lr'] = learning_rate
    group['tag'] = "lm_head"
    optimizer_grouped_parameters.append(group)
    return optimizer_grouped_parameters


def get_unfreezing_funcs(optimizer_grouped_parameters, warmup_portion, total_steps, use_unfreezing=True):
    def get_lr_lambda(block, use_unfreezing):
        if use_unfreezing:
            NUM_STAGES = 50
            warmup_start = int((24 - block) * total_steps / NUM_STAGES)
            warmup_length = int((total_steps - warmup_start) * warmup_portion)
        else:
            warmup_start = 0
            warmup_length = int((total_steps - warmup_portion) * warmup_portion)
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
            lm_funcs.append(get_lr_lambda(24, use_unfreezing))
        else:
            try:
                block = int(tag[2:-1])  # get the block number of a parameters group
                lm_funcs.append(get_lr_lambda(block, use_unfreezing))
            except:
                print("Exists invalid block numbers while creating unfreezing scheme")
    return lm_funcs


def construct_grouped_parameters(param_optimizer, learning_rate, use_discr=True):
    # discriminative lr
    optimizer_grouped_parameters = []
    weight_decay = 0.01
    lr_decay_rate =0.9
    num_blocks = 24
    # no_decay = ['bias', 'ln_']
    no_decay = ['bias', 'LayerNorm.weight']
    need_decay = lambda n: not any(nd in n for nd in no_decay)
    in_block = lambda name, i: "h.{}.".format(i) in name

    if use_discr:
        embedding_lr = learning_rate * (lr_decay_rate ** num_blocks)
    else:
        embedding_lr = learning_rate
    group = {'params': [p for n, p in param_optimizer if 'wte' in n],
             'lr': embedding_lr, 'tag': "wte", 'weight_decay': weight_decay}
    optimizer_grouped_parameters.append(group)

    group = {'params': [p for n, p in param_optimizer if 'wpe' in n],
             'lr': embedding_lr, 'tag': "wpe", 'weight_decay': weight_decay}
    optimizer_grouped_parameters.append(group)

    for i in range(num_blocks):
        tag = "h.{}.".format(i)
        if use_discr:
            lr = learning_rate * (lr_decay_rate ** (num_blocks - i))
        else:
            lr = learning_rate
        group = {'lr': lr, 'tag': tag, 'weight_decay': weight_decay}
        group['params'] = [p for n, p in param_optimizer if need_decay(n) and in_block(n, i)]
        optimizer_grouped_parameters.append(group)

        group = {'lr': lr, 'tag': tag, 'weight_decay': 0}
        group['params'] = [p for n, p in param_optimizer if not need_decay(n) and in_block(n, i)]
        optimizer_grouped_parameters.append(group)

    group = {'params': [p for n, p in param_optimizer if "ln_f.weight" in n],
             'lr': learning_rate, 'tag': "ln_f", 'weight_decay': weight_decay}
    optimizer_grouped_parameters.append(group)

    group = {'params': [p for n, p in param_optimizer if "ln_f.bias" in n],
             'lr': learning_rate, 'tag': "ln_f", 'weight_decay': 0}
    optimizer_grouped_parameters.append(group)

    return optimizer_grouped_parameters


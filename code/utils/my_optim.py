def construct_grouped_parameters(param_optimizer, lr):
    # discriminative lr
    optimizer_grouped_parameters = []
    all_name = []
    no_decay = ['bias', 'ln_']
    need_decay = lambda n: not any(nd in n for nd in no_decay)
    in_block = lambda name, i: "h.{}.".format(i) in name

    num_blocks = 12
    for i in range(num_blocks):
        group = {}
        group['params'] = [p for n, p in param_optimizer if need_decay(n) and in_block(n, i)]
        group['weight_decay'] = 0.01
        group['tag'] = "h.{}.".format(i)
        group['lr'] = lr * (0.5 ** (11 - i))  # soft mask
        optimizer_grouped_parameters.append(group)

        group = {}
        group['params'] = [p for n, p in param_optimizer if not need_decay(n) and in_block(n, i)]
        group['weight_decay'] = 0
        group['tag'] = "h.{}.".format(i)
        group['lr'] = lr * (0.5 ** (11 - i))  # soft mask
        optimizer_grouped_parameters.append(group)

    group = {}
    group['params'] = [p for n, p in param_optimizer if "ln_f" in n]
    group['weight_decay'] = 0
    group['lr'] = lr
    group['tag'] = "ln_f"
    optimizer_grouped_parameters.append(group)

    group = {}
    group['params'] = [p for n, p in param_optimizer if n == 'weight']
    group['weight_decay'] = 0.01
    group['lr'] = lr
    group['tag'] = "lm_head"
    optimizer_grouped_parameters.append(group)
    return optimizer_grouped_parameters


def unfreezing_parameters(optimizer_grouped_parameters, epoch):
    if epoch > 12:
        return
    for group in optimizer_grouped_parameters:
        if group['tag'] == "h.{}.".format(12 - epoch):
            group['lr'] *= 1e20

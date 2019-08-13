import csv

def get_values_lexicon(path):
    values_dict = {}
    with open(path, newline='\n') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if row[1] not in values_dict:
                values_dict[row[1]]=[]
            values_dict[row[1]].append(row[0])
    return values_dict

def values_lexicon_encode(path,tokenizer):
    values_dict = get_values_lexicon(path)
    values_dict_filt = {}
    values_dict_filt_inv = {}
    for k,v in values_dict.items():
        for word in v:
            encoded = tokenizer.encode(word)
            if len(encoded)==1:
                word = encoded[0]
                if k not in values_dict_filt:
                    values_dict_filt[k]=[]
                values_dict_filt[k].append(word)
                values_dict_filt_inv[word] = k
    return values_dict_filt, values_dict_filt_inv
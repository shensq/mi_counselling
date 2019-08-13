import sys
sys.path.append('/data/chuancen/pip_package')
sys.path.append('..')
import nltk
from nltk.translate.meteor_score import meteor_score
from utils import get_values_lexicon,values_lexicon_encode
from pytorch_transformers import GPT2Tokenizer

def main():
    nltk.data.path.append('/data/chuancen/pip_package/nltk_data')
    print(nltk.__version__)
    file_handler = open('../../result/reference_SR_only.txt','r')
    ref = file_handler.readlines()
    file_handler = open('../../result/SR_only.txt','r')
    hyp = file_handler.readlines()

    print("#ref{} #hyp{}".format(len(ref),len(hyp)))
    meteor_sum = 0
    for i in range(min(len(ref),len(hyp))):
        meteor_sum += meteor_score([ref[i]],hyp[i])

    meteor_sum/=min(len(ref),len(hyp))
    print(meteor_sum)
    
    tokenizer = GPT2Tokenizer.from_pretrained('/data/chuancen/LIT/models/345M_Alex')


if __name__ == "__main__":
    main()
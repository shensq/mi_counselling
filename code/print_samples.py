import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, default='345M_Alex')
    args = parser.parse_args()
    print(args)
    f_ref = open('../result/reference_'+args.filepath+'.txt','r')
    f_hyp = open('../result/'+args.filepath+'.txt','r')

    ref = f_ref.readlines()
    hyp = f_hyp.readlines()
    assert len(ref)==len(hyp)
    for i in range(len(ref)):
        print("="*10+"sample {}".format(i)+"="*10)
        print(ref[i])
        print(hyp[i])


if __name__ == "__main__":
    main()
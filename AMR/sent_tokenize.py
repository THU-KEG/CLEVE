import nltk
import argparse
import os
from random import shuffle
from tqdm import tqdm

def sent_tokenize(args):
    all_sents = []

    data_dir = args.data_dir
    years = os.listdir(data_dir)
    years = [data_dir+'/'+year for year in years]
    for year in tqdm(years, desc='years'):
        subfolders = os.listdir(year)
        if year in ['1987','1988','1989','1990']:
            continue
        subfolders = [year+'/'+subfolder for subfolder in subfolders]
        for subfolder in tqdm(subfolders,desc='subfolders'):
            subsubfolders = os.listdir(subfolder)
            subsubfolders = [subfolder+'/'+subsubfolder for subsubfolder in subsubfolders]
            for subsubfolder in subsubfolders:
                files = os.listdir(subsubfolder)
                files = [subsubfolder+'/'+file for file in files]
                for file in files:
                    with open(file,'r') as f:
                        text = f.read()
                        sents = nltk.sent_tokenize(text)
                        sents = [sent for sent in sents if len(nltk.word_tokenize(sent))>=10 and len(nltk.word_tokenize(sent))<50]
                        all_sents.extend(sents)

    shuffle(all_sents)
    all_sents = all_sents[-args.num:]
    with open("nyt_sent_limit.txt",'w') as f:
        for sent in all_sents:
            f.write(sent)
            f.write('\n')



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir.",
    )
    parser.add_argument(
        "--num",
        default=None,
        type=int,
        required=True,
        help="number of sentences we used",
    )
    args = parser.parse_args()
    sent_tokenize(args)
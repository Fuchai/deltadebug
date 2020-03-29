import re
import string

import pandas as pd

files=["alternate_12-03-17-11-37.txt","puctdirichletforkalternate_12-04-23-35-50.txt"]

def alphabets_and_numbers(line):
    words=line.split()
    values={}
    table_id=""
    phrase=""
    model_name=None
    for word_idx, word in enumerate(words):
        if word_idx==0:
            model_name=word
        else:
            if any(c.isalpha() for c in word):
                word=re.sub(r'[^\w\s]', '', word.strip())
                phrase+=(word+" ")
            else:
                if word[-1]==".":
                    word=word[:-1]
                if word[-1]==",":
                    word=word[:-1]
                try:
                    num=int(word)
                except ValueError:
                    num=float(word)
                values[phrase]=num
                table_id+=phrase
                phrase=""
    return model_name, table_id, values

def process_log(file):
    dfs={}
    # for file in files:
    with open(file, 'r') as f:
        for line in f:
            mn1, ti, values=alphabets_and_numbers(line)
            if ti in dfs:
                df_dict=dfs[ti]
                for key, val in values.items():
                    df_dict[key].append(val)
            else:
                newd={}
                for key, val in values.items():
                    newd[key]=[val]
                dfs[ti]=newd

    name_map={"train epoch resampling running value loss running policy loss running p diff ":"train",
              "valid epoch resampling validation value loss validation policy loss validation p diff ":"valid"}
    pddfs={}
    for df in dfs.keys():
        pddfs[name_map[df]]=pd.DataFrame.from_dict(dfs[df])
    return pddfs


if __name__ == '__main__':
    # alphabets_and_numbers("highpuctshortgame28 train epoch   28, resampling  170. running value loss: 0.43749. running policy loss: 1.16325. running p diff: 0.22115")
    pddfs=process_log("log/alternate_12-03-17-11-37.txt")
    for name, df in pddfs.items():
        df.to_csv("csvs/"+name+".csv", index=False)
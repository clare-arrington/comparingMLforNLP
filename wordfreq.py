import math
import pandas as pd
from collections import Counter

wine = pd.read_csv('wine.csv')

mid = wine[wine.binary_label == 0]
top = wine[wine.binary_label == 1]

mid_blurb = " ".join(mid.desc)
top_blurb = " ".join(top.desc)


mid_freq = Counter(mid_blurb.split(" "))
mid_total = sum(mid_freq.values())
top_freq = Counter(top_blurb.split(" "))
top_total = sum(top_freq.values())


def compare(n_words):
    word_list = []
    LL_list = []
    corp_list = []
    
    corpora1 = mid_freq.most_common(n_words)
    corpora2 = top_freq
    
    for word_freq in corpora1:
        if (word_freq[1] / mid_total > 
            corpora2[word_freq[0]] / top_total):
            more_freq = 'Mid'
        else:
            more_freq = 'Top'
            
        word_list.append(word_freq[0])
        LL_list.append(LL(word_freq[1], corpora2[word_freq[0]]))
        corp_list.append(more_freq)
        
    return pd.DataFrame({'word': word_list, 'LL': LL_list, 'Corpus': corp_list})
            


def LL(obs1, obs2):
    sumObs = obs1 + obs2
    sumCorp = mid_total + top_total
    
    E1 = (mid_total * sumObs) / sumCorp
    E2 = (top_total * sumObs) / sumCorp
    
    G2 = 2 * ((obs1 * math.log(obs1 / E1)) + (obs2 * math.log(obs2 / E2)))
    
    return G2

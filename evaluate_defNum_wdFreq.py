import json
from evaluate import evaluate
import numpy as np

target_words = set(json.load(open('../data_from_dujiaju/target_words.json')))
#word_freq = {}
wf_ind = {}
for line in open('word_freq.txt').readlines():
    #word_freq[line.split()[0]] = int(line.split()[1])
    if line.split()[0] in target_words:
        if int(line.split()[1])>=5000:
            wf_ind[line.split()[0]] = 0
        elif int(line.split()[1])>=500:
            wf_ind[line.split()[0]] = 1
        elif int(line.split()[1])>=50:
            wf_ind[line.split()[0]] = 2
        else:
            wf_ind[line.split()[0]] = 3
            
dictionary = json.load(open('../data_from_dujiaju/dictionary_multisense.json'))
dn_ind = {} # defi num
for wd, value in dictionary.items():
    if wd in target_words:
        dn_ind[wd] = len(value)
            
score, reference, pred_all = [[],[],[]]
for i in range(10):
    #sc = json.load(open(str(i)+'_score_list.json'))
    ref = json.load(open(str(i)+'_reference.json'))
    pre = json.load(open(str(i)+'_pred_all.json'))
    #score.extend(sc)     
    reference.extend(ref)
    pred_all.extend(pre)
    print(str(i)+'_json readed.')

length = len(target_words)
assert len(reference) == length

map_freq = [[],[],[],[]]
map_dn = [[],[],[],[],[],[]]
map = []

target_words0 = []
for i in range(10):
    target_words0.extend((open(str(i)+'target_words.txt', 'r').readlines()[0]).split())
for i in range(length):
    m = evaluate(reference[i], pred_all[i])
    map.append(m)
    map_freq[wf_ind[target_words0[i]]].append(m)
    if dn_ind[target_words0[i]]>5:
        map_dn[5].append(m)
    else:
        map_dn[dn_ind[target_words0[i]]-1].append(m)
print('MAP in each word frequences:')
for m in map_freq:
    print(np.mean(np.array(m)))
print('MAP in each defi numbers:')
for m in map_dn:
    print(np.mean(np.array(m)))
print('-----MAP: ', np.mean(np.array(map)))
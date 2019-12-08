def evaluate(ground_truth, prediction):
    index = 1
    correct = 0
    point = 0
    for predicted_sememe in prediction:
        if predicted_sememe in ground_truth:
            correct += 1
            point += (correct / index)
        index += 1
    point /= len(ground_truth)
    return point
    
def heursure(score):
    '''
    score_diff = score.copy()
    for i in range(40,0,-1): # max num of sememes is 41
        score_diff[i] = score_diff[i-1] - score_diff[i]
    score_diff[0] = 0.
    index_cut = score_diff.index(max(score_diff))
    '''
    for i, sc in enumerate(score):
        if sc<0:
            index_cut = i
            break
    if index_cut==0:
        index_cut=1
    return index_cut
    
def evaluateF1(ground_truth, prediction):
    total_right = 0.
    total_ref = 0.
    total_can = 0.
    for r,c in zip(ground_truth, prediction):
        right = set.intersection(set(r), set(c))
        total_right += len(right)
        total_ref += len(r)
        total_can += len(c)
    total_can = total_can if total_can != 0 else 1
    precision = total_right/float(total_can)*100.
    if total_ref == 0:
        recall = 0
        print(ground_truth,prediction)
    else:
        recall = total_right/float(total_ref)*100.
    if precision == 0 or recall == 0:
        F1 = 0.
    else:
        F1 = precision*recall*2./(precision+recall)
    return precision, recall, F1
    
def get_prediction(score):
    # score: (batch_num * batch_size(or less), sememe_num) -> (38 * 128, 1400)
    pred_sememe_all = list()
    for score_batch in score:
        for sc in score_batch:
            result = list()
            for i,s in enumerate(sc):
                result.append((i,s))
            result.sort(key=lambda result: result[1], reverse=True)
            score_sort = list()
            pred_sememe = list()
            for pred, s_s in result:
                pred_sememe.append(pred)
                score_sort.append(s_s)
            index_cut = heursure(score_sort)
            pred_sememe = pred_sememe[:index_cut]
            pred_sememe_all.append(pred_sememe)
    return pred_sememe_all
    
'''
import numpy as np
import pickle

path = './'
reference = pickle.load(open(path+'reference.pkl', 'rb'))
score_list = pickle.load(open(path+'score_list.pkl', 'rb'))
print(evaluateF1(reference, get_prediction(score_list)))
'''
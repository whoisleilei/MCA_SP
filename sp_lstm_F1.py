import argparse, json, os, random, sys, torch
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm
import data
from data import SPDataset, sp_collate_fn, load_data, build_word2sememe
from evaluate import evaluate, heursure, evaluateF1
from model import SPLSTM
from utils import cache, data_path, device, output_path, private_path 

def main(epoch_num, verbose, fold):
    word2index, index2word, word2vec, sememe2index, index2sememe, word_sememe_idx, word_sememe_test_idx = load_data(fold)
    data.sememe_number = len(index2sememe)
    random.shuffle(word_sememe_idx)
    length = len(word_sememe_idx)
    train_dataset = SPDataset(word_sememe_idx[:int(0.9*length)])
    valid_dataset = SPDataset(word_sememe_idx[int(0.9*length):])
    test_dataset = SPDataset(word_sememe_test_idx)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=sp_collate_fn)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=128, shuffle=True, collate_fn=sp_collate_fn)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=sp_collate_fn)
    train_word2sememe = build_word2sememe(train_dataset, len(word2index), len(index2sememe))
    model = SPLSTM(len(word2index), word2vec.shape[1], len(index2sememe), 256, 1, data.sememe_number, train_word2sememe)
    model.embedding.weight.data = torch.from_numpy(word2vec)
    model.to(device)
    sparse_parameters_name = ['embedding.weight', 'sememe_embedding.embedding.weight']
    sparse_parameters = [para for name, para in model.named_parameters() if name in sparse_parameters_name]
    non_sparse_parameters = [para for name, para in model.named_parameters() if name not in sparse_parameters_name]
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, non_sparse_parameters), lr=0.001)
    sparse_optimizer = torch.optim.SparseAdam(filter(lambda p: p.requires_grad, sparse_parameters), lr=0.001)
    best_valid_map = 0 #####
    best_test_map = 0 #####
    for epoch in range(epoch_num):
        torch.cuda.empty_cache()
        print('epoch', epoch)
        model.train()
        train_map = 0
        train_loss = 0
        for words_t, sememes_t, definition_words_t, sememes in tqdm(train_dataloader, disable=verbose):
            optimizer.zero_grad()
            sparse_optimizer.zero_grad()
            loss, score, indices = model('train', x=definition_words_t, y=sememes_t, w=words_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            sparse_optimizer.step()
            predicted = indices.detach().cpu().numpy().tolist()
            for i in range(len(sememes)):
                train_map += evaluate(sememes[i], predicted[i])
            train_loss += loss.item()
        print('train loss: ', train_loss)
        print('train map: %.2f'%(train_map*100 / len(train_dataset)))
        model.eval()
        with torch.no_grad():
            valid_map = 0
            for words_t, sememes_t, definition_words_t, sememes in tqdm(valid_dataloader, disable=verbose):
                loss, _, indices = model('train', x=definition_words_t, y=sememes_t, w=words_t)
                predicted = indices.detach().cpu().numpy().tolist()
                for i in range(len(sememes)):
                    m = evaluate(sememes[i], predicted[i])
                    valid_map += m
            print('valid map: %.2f'%(valid_map*100 / len(valid_dataset)))
            if valid_map > best_valid_map:
                print('-----best_valid_map-----')
                best_valid_map = valid_map
                test_map = 0
                score_list = list() #####
                pred_all = list() #####
                reference = list() #####
                for words_t, sememes_t, definition_words_t, sememes in tqdm(test_dataloader, disable=verbose):
                    loss, score, indices = model('train', x=definition_words_t, y=sememes_t, w=words_t)
                    predicted = indices.detach().cpu().numpy().tolist()
                    pred_all.extend(predicted) #####
                    for i in range(len(sememes)):
                        test_map += evaluate(sememes[i], predicted[i])
                    score_list.extend(np.around(score.detach().cpu().numpy(), 3).tolist()) #####
                    reference.extend(sememes) #####
                if epoch>20:
                    #json.dump(score_list, open(str(fold)+'_score_list.json', 'w'))
                    json.dump(pred_all, open(str(fold)+'_pred_all.json', 'w'))
                    json.dump(reference, open(str(fold)+'_reference.json', 'w'))
                best_test_map = test_map
                candidate = list()
                for i in range(len(score_list)):
                    score_sorted = sorted(score_list[i], reverse=True)
                    candidate.append(pred_all[i][:heursure(score_sorted)])
                print('precision, recall, F1: %.2f %.2f %.2f'%(evaluateF1(reference, candidate)))
                print('test map: %.2f'%(test_map*100 / len(test_dataset)))
    print('best test map: %.2f'%(best_test_map*100 / len(test_dataset))) #####

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epoch_num', type=int, default=30)
    parser.add_argument('-v', '--verbose',default=True, action='store_false')
    parser.add_argument('-g', '--gpu', type=str, default='0')
    parser.add_argument('-f', '--fold', type=int, default=0)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args.epoch_num, args.verbose, args.fold)
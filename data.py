import json
import os

import numpy as np
import torch
import torch.utils.data

from utils import private_path, cache, data_path, device


sememe_number = 0


class SPDataset(torch.utils.data.Dataset):
    def __init__(self, instances):
        self.instances = instances

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        return self.instances[index]


#@cache(['word2index.json', 'index2word.json', 'word2vec.npy', 'sememe2index.json', 'index2sememe.json', 'word_sememe.json', 'word_sememe_idx.json'], private_path)
def load_data(i):
    hownet = json.load(open(os.path.join(data_path, 'hownet.json')))
    sememe_list = json.load(open(os.path.join(data_path, 'sememe.json')))
    word_index = json.load(open(os.path.join(data_path, 'word_index.json')))
    index_word = json.load(open(os.path.join(data_path, 'index_word.json')))
    word_vector = np.load(os.path.join(data_path, 'word_vector.npy'))
    dictionary = json.load(open(os.path.join(data_path, 'dictionary.json')))
    target_words = json.load(open(os.path.join(data_path, 'target_words.json')))
    length = int(len(target_words)/10)
    if i == 9:
        test_words = target_words[i*length:]
    else:
        test_words = target_words[i*length:(i+1)*length]
    print('dictionary len:', len(dictionary))
    print('hownet len:', len(hownet))
    word_sememe = list()
    for word, senses in hownet.items(): # 最终存储的数据word_sememe_idx顺序与HowNet中的一致
        if word not in word_index:
            continue
        if word not in dictionary:
            continue
        sememe_set = set()
        for sense in senses:
            for sememe in sense:
                if sememe in sememe_list:
                    sememe_set.add(sememe)
        if len(sememe_set) > 0:
            definition_words = [word, '：']     # using word prefix
            #definition_words = []          # not using word prefix
            definition_words.extend(dictionary[word]['definition_words'])
            word_sememe.append({'word': word, 'sememes': list(sememe_set), 'definition_words': definition_words})
    print('hownet and word with definition and word with vector', len(word_sememe))
    word_set = set()
    for instance in word_sememe:
        word_set.add(instance['word'])
    word_set.add('：')
    for word, value in dictionary.items():
        definition_words = value['definition_words']
        for def_word in definition_words:
            if def_word in word_index:
                word_set.add(def_word)
    word2index = dict()
    index2word = list()
    word2index['<padding>'] = 0
    index2word.append('<padding>')
    word2index['<oov>'] = 1
    index2word.append('<oov>')
    word2vec = np.zeros((len(word_set) + 2, word_vector.shape[1]), dtype=np.float32)
    for word in word_set:
        index = len(word2index)
        word2index[word] = index
        index2word.append(word)
        vec = word_vector[word_index[word]]
        word2vec[index, :] = vec
    sememe2index = dict()
    index2sememe = list()
    for sememe in sememe_list:
        sememe2index[sememe] = len(sememe2index)
        index2sememe.append(sememe)
    word_sememe_idx = list()
    word_sememe_test_idx = list()
    #f = open(str(i)+'target_words.txt', 'w')
    for instance in word_sememe:
        word = instance['word']
        sememes = instance['sememes']
        definition_words = instance['definition_words']
        word_idx = word2index[word]
        sememe_idx = [sememe2index[sememe] for sememe in sememes]
        def_word_idx = list()
        for def_word in definition_words:
            if def_word in word2index:
                def_word_idx.append(word2index[def_word])
            else:
                def_word_idx.append(word2index['<oov>'])
        if word in test_words:
            #f.write(word+' ')
            word_sememe_test_idx.append({'word': word_idx, 'sememes': sememe_idx, 'definition_words': def_word_idx})
        else:
            word_sememe_idx.append({'word': word_idx, 'sememes': sememe_idx, 'definition_words': def_word_idx})
    #f.close()
    #exit()
    return word2index, index2word, word2vec, sememe2index, index2sememe, word_sememe_idx, word_sememe_test_idx


def build_word2sememe(train_dataset, word_number, sememe_number):
    max_sememe_number = max([len(instance['sememes']) for instance in train_dataset])
    r = np.zeros((word_number, max_sememe_number), dtype=np.int64)
    r.fill(sememe_number)
    for instance in train_dataset:
        word_idx = instance['word']
        sememes = instance['sememes']
        r[word_idx, 0:len(sememes)] = np.array(sememes)
    r = torch.tensor(r, dtype=torch.int64, device=device)
    return r


def build_sentence_numpy(sentences):
    max_length = max([len(sentence) for sentence in sentences])
    sentence_numpy = np.zeros((len(sentences), max_length), dtype=np.int64)
    for i in range(len(sentences)):
        sentence_numpy[i, 0:len(sentences[i])] = np.array(sentences[i])
    return sentence_numpy


def get_sememe_label(sememes, sememe_number):
    l = np.zeros((len(sememes), sememe_number), dtype=np.float32)
    for i in range(len(sememes)):
        for s in sememes[i]:
            l[i, s] = 1
    return l


def sp_collate_fn(batch):
    words = [instance['word'] for instance in batch]
    sememes = [instance['sememes'] for instance in batch]
    definition_words = [instance['definition_words'] for instance in batch]
    words_t = torch.tensor(np.array(words), dtype=torch.int64, device=device)
    sememes_t = torch.tensor(get_sememe_label(sememes, sememe_number), dtype=torch.float32, device=device)
    definition_words_t = torch.tensor(build_sentence_numpy(definition_words), dtype=torch.int64, device=device)
    return words_t, sememes_t, definition_words_t, sememes
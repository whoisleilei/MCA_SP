import torch

from utils import device


class BiSentenceLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1) -> None:
        super().__init__()
        if num_layers == 1:
            self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        else:
            self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, dropout=0.5, batch_first=True, bidirectional=True)

    def forward(self, x, x_len):
        # x: Tensor(batch, length, input_size) float32
        # x_len: Tensor(batch) int64
        # x_len_sort_idx: Tensor(batch) int64
        _, x_len_sort_idx = torch.sort(-x_len)
        # x_len_sort_idx: Tensor(batch) int64
        _, x_len_unsort_idx = torch.sort(x_len_sort_idx)
        x = x[x_len_sort_idx]
        x_len = x_len[x_len_sort_idx]
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=True)
        # ht: Tensor(num_layers * 2, batch, hidden_size) float32
        # ct: Tensor(num_layers * 2, batch, hidden_size) float32
        h_packed, (ht, ct) = self.lstm(x_packed, None)
        ht = ht[:, x_len_unsort_idx, :]
        ct = ct[:, x_len_unsort_idx, :]
        # h: Tensor(batch, length, hidden_size * 2) float32
        h, _ = torch.nn.utils.rnn.pad_packed_sequence(h_packed, batch_first=True)
        h = h[x_len_unsort_idx]
        return h, (ht, ct)


class SememeEmbedding(torch.nn.Module):
    def __init__(self, sememe_number, embedding_dim):
        super().__init__()
        self.sememe_number = sememe_number
        self.embedding_dim = embedding_dim
        self.embedding = torch.nn.Embedding(self.sememe_number + 1, self.embedding_dim, padding_idx=self.sememe_number, max_norm=5, sparse=True)
        self.embedding.weight.data[self.sememe_number] = 0

    def forward(self, x):
        # x: T(batch_size, max_word_number, max_sememe_number) padding: self.sememe_number
        # x_mask: T(batch_size, max_word_number, max_sememe_number)
        x_mask = torch.lt(x, self.sememe_number).to(torch.float32)
        # x_embedding: T(batch_size, max_word_number, max_sememe_number, embedding_dim)
        x_embedding = self.embedding(x)
        # x_average: T(batch_size, max_word_number, embedding_dim)
        x_average = torch.sum(x_embedding, dim=2) / torch.max(torch.sum(x_mask, dim=2, keepdim=True), torch.tensor([[[1]]], device=device, dtype=torch.float32))
        return x_average


class SPLSTM(torch.nn.Module):
    def __init__(self, vocabulary_size, embedding_dim, sememe_number, hidden_size, lstm_layers, class_number, train_word2sememe):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.sememe_number = sememe_number
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.class_number = class_number
        self.embedding = torch.nn.Embedding(self.vocabulary_size, self.embedding_dim, padding_idx=0, max_norm=5, sparse=True)
        self.embedding.weight.requires_grad = False
        self.sememe_embedding = SememeEmbedding(self.sememe_number, self.embedding_dim)
        self.embedding_dropout = torch.nn.Dropout()
        self.lstmencoder = BiSentenceLSTM(self.embedding_dim, self.hidden_size, self.lstm_layers)
        self.fc = torch.nn.Linear(self.hidden_size * 2 + self.embedding_dim, self.class_number)
        self.loss = torch.nn.MultiLabelSoftMarginLoss()
        self.train_word2sememe = train_word2sememe

    def forward(self, operation, x=None, y=None, w=None):
        # x: T(batch_size, max_word_number) 0表示占位符
        # y: T(batch_size, class_number) 输入每个样例对应的sememe序号. 用-1表示已经输入完毕
        # x_word_embedding: T(batch_size, max_word_number, embedding_dim)
        x_word_embedding = self.embedding(x)
        # x_sememe: T(batch_size, max_word_number, max_sememe_number)
        x_sememe = self.train_word2sememe[x]
        x_sememe[:, 0, :] = self.sememe_number
        # x_sememe_embedding: T(batch_size, max_word_number, embedding_dim)
        x_sememe_embedding = self.sememe_embedding(x_sememe)
        # x_embedding: T(batch_size, max_word_number, max_sememe_number)
        x_embedding = x_word_embedding + x_sememe_embedding
        x_embedding = self.embedding_dropout(x_embedding)
        # mask: T(batch_size, max_word_number)
        mask = torch.gt(x, 0).to(torch.int64)
        # x_len: T(batch_size)
        x_len = torch.sum(mask, dim=1)
        # h: T(batch_size, max_word_number, hidden_size * 2)
        h, (ht, _) = self.lstmencoder(x_embedding, x_len)
        # ht: T(bat, hid*2)
        ht = torch.transpose(ht[ht.shape[0] - 2:, :, :], 0, 1).contiguous().view(x_len.shape[0], self.hidden_size*2)
        # alpha: T(bat, max_word_num, 1)
        alpha = (h.bmm(ht.unsqueeze(2)))
        #vd, _ = torch.max(h, dim=1)
        vd = torch.sum(h*alpha, 1)
        vd = torch.cat((vd, self.embedding(w)), 1)
        # score: T(batch_size, sememe_number)
        score = self.fc(vd)
        _, indices = torch.sort(score, descending=True)
        if operation == 'train':
            loss = self.loss(score, y)
            return loss, score, indices
        elif operation == 'inference':
            return score, indices
        
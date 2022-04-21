import gensim
import torch
import torch.nn as nn
import re
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader


class CnnWord2Vec(nn.Module):
    def __init__(self, embedding_size=200, number_of_classes=3, batch_size=50):
        super(CnnWord2Vec, self).__init__()
        self.batch_size = batch_size
        self.number_of_classes = number_of_classes
        self.input_channel = 1
        self.embedding_size = embedding_size

        self.convolution_layer_3dfilter = nn.Conv2d(self.input_channel, 100, (3, self.embedding_size))
        nn.init.xavier_uniform_(self.convolution_layer_3dfilter.weight)

        self.convolution_layer_4dfilter = nn.Conv1d(self.input_channel, 100, (4, self.embedding_size))
        nn.init.xavier_uniform_(self.convolution_layer_4dfilter.weight)

        self.convolution_layer_5dfilter = nn.Conv1d(self.input_channel, 100, (5, self.embedding_size))
        nn.init.xavier_uniform_(self.convolution_layer_4dfilter.weight)

        self.dropout = nn.Dropout(p=0.5)

        self.linear = nn.Linear(300, self.number_of_classes)
        nn.init.xavier_uniform_(self.linear.weight)

        self.softmax = nn.Softmax(dim=1)


    def forward(self, inn):
        embedded = inn
        embedded = embedded.unsqueeze(1)

        conv_opt3 = self.convolution_layer_3dfilter(embedded)
        conv_opt4 = self.convolution_layer_4dfilter(embedded)
        conv_opt5 = self.convolution_layer_5dfilter(embedded)

        conv_opt3 = nn.functional.relu(conv_opt3).squeeze(3)
        conv_opt4 = nn.functional.relu(conv_opt4).squeeze(3)
        conv_opt5 = nn.functional.relu(conv_opt5).squeeze(3)

        conv_opt3 = nn.functional.max_pool1d(conv_opt3, conv_opt3.size(2)).squeeze(2)
        conv_opt4 = nn.functional.max_pool1d(conv_opt4, conv_opt4.size(2)).squeeze(2)
        conv_opt5 = nn.functional.max_pool1d(conv_opt5, conv_opt5.size(2)).squeeze(2)

        conv_opt = torch.cat((conv_opt3, conv_opt4, conv_opt5), 1)
        conv_opt = self.dropout(conv_opt)

        linear_opt = self.linear(conv_opt)

        return linear_opt

class WordDataset(Dataset):
    def __init__(self, txt_file='./Sentences_50Agree.txt', embedding_file='./sim.expand.200d.bin'):
        self.embed_model = gensim.models.KeyedVectors.load_word2vec_format(embedding_file, binary=True)
        self.class_dic = {'positive': '0', 'negative': '1', 'neutral': '2'}
        self.data = open(txt_file, 'r', errors='ignore')
        self.data_r = self.data.readlines()
        self.texts = [e.strip('\n').split('@')[0] for e in self.data_r]
        self.labels = [self.class_dic[e.strip('\n').split('@')[1]] for e in self.data_r]
        self.max_length = 50

    def __len__(self):
        return len(self.texts)

    def clean_str(self, sent):
        sent = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sent)
        sent = re.sub(r"\'s", " \'s", sent)
        sent = re.sub(r"\'ve", " \'ve", sent)
        sent = re.sub(r"n\'t", " n\'t", sent)
        sent = re.sub(r"\'re", " \'re", sent)
        sent = re.sub(r"\'d", " \'d", sent)
        sent = re.sub(r"\'ll", " \'ll", sent)
        sent = re.sub(r",", " , ", sent)
        sent = re.sub(r"!", " ! ", sent)
        sent = re.sub(r"\(", " ( ", sent)
        sent = re.sub(r"\)", " ) ", sent)
        sent = re.sub(r"\?", " ? ", sent)
        sent = re.sub(r"\s{2,}", " ", sent)
        return sent.strip().lower()

    def __getitem__(self, idx):
        text_tokenized = self.clean_str(self.texts[idx]).split()
        text_embeddings = [torch.Tensor(self.embed_model.get_vector(e)).unsqueeze(0) for e in text_tokenized if self.embed_model.has_index_for(e)]
        if len(text_embeddings) <=  self.max_length:
            for i in range(self.max_length - len(text_embeddings)):
                text_embeddings.append(torch.Tensor(self.embed_model.get_vector('pad')).unsqueeze(0))
        text_cat = torch.cat([e for e in text_embeddings], dim=0)
        label = torch.LongTensor([float(self.labels[idx])])

        return text_cat, label


if __name__ == "__main__":
    model = CnnWord2Vec().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters())

    dataset = WordDataset()
    dataloader = DataLoader(dataset = dataset, batch_size = 50, shuffle = True)

    for epoch in range(50):
        running_loss = 0
        for batch in dataloader:
            text, label = batch
            text = text.cuda()
            label = label.squeeze(1).cuda()

            optimizer.zero_grad()
            out = model(text)
            loss = criterion(out, label)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(running_loss)

    torch.save(model.state_dict(), './cnn2vec.pt')
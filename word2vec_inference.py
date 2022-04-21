import os
import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
import re
import gensim
from word2vec_analysis import CnnWord2Vec
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True)
args = parser.parse_args()

word2vec_list = open('./word2vec_result_210.txt', 'a')
word2vec_list.write('CIK' + '\t' + 'REPORT PERIOD END DATE' + '\t' + 'FILE DATE' + '\t' + 'FORM TYPE' + '\t' + 'POS' + '\t' + 'NEG' + '\t' + 'NEU' + '\n')

PATH = './cnn2vec.pt'
model = CnnWord2Vec()
model.load_state_dict(torch.load(PATH))
model.eval()
model.cuda()

word2vec = gensim.models.KeyedVectors.load_word2vec_format('./sim.expand.200d.bin', binary=True)
softmax = nn.Softmax(dim=1)

count = 0
for txt in os.listdir(args.data_dir):
        if '.txt' in txt and 'download' not in txt and 'DOWNLOADLOG' not in txt and 'newfile' not in txt and 'temp' not in txt and '._' not in txt:
            count += 1
            if count >= 17200 and count < 18000:
                pos = 0
                neg = 0
                neu = 0
                var_dic = {}
                fl = open(os.path.join(args.data_dir, txt), errors='ignore')
                fl_read = fl.readlines()
                for fr in fl_read:
                    if '<HEADER>'in fr or '</HEADER>' in fr or 'COMPANY NAME' in fr or 'CIK' in fr or 'SIC' in fr or 'FORM TYPE' in fr or 'REPORT PERIOD END DATE' in fr or 'FILE DATE' in fr or '<SECTION>' in fr or '</SECTION>' in fr:
                        if 'CIK' in fr:
                            var_dic['CIK'] = fr.strip('CIK: ').strip('\n')
                        if 'REPORT PERIOD END DATE' in fr:
                            var_dic['REPORT PERIOD END DATE'] = fr.strip('REPORT PERIOD END DATE: ').strip('\n')
                        if 'FILE DATE' in fr:
                            var_dic['FILE DATE'] = fr.strip('FILE DATE: ').strip('\n')
                        if 'FORM TYPE' in fr:
                            var_dic['FORM TYPE'] = fr.strip('FORM TYPE: ').strip('\n')
                    else:
                        lines = nltk.sent_tokenize(fr)
                        for sent in lines:
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
                            sent_tokenized = sent.split()
                            emb = [word2vec[e] for e in sent_tokenized if word2vec.has_index_for(e)]

                            if len(emb) <  50:
                                for i in range(50 - len(emb)):
                                    emb.append(word2vec.get_vector('pad'))
                                emb_t = [torch.Tensor(e).unsqueeze(0) for e in emb]
                                emb_cat = torch.cat(emb_t, dim=0).unsqueeze(0).cuda()
                                out = model(emb_cat)
                                out_softmax = F.softmax(out).tolist()

                                pos += out_softmax[0][0]
                                neg += out_softmax[0][1]
                                neu += out_softmax[0][2]

                            elif len(emb) == 50:
                                emb_t = [torch.Tensor(e).unsqueeze(0) for e in emb]
                                emb_cat = torch.cat(emb_t, dim=0).unsqueeze(0).cuda()
                                out = model(emb_cat)
                                out_softmax = F.softmax(out).tolist()
                                pos += out_softmax[0][0]
                                neg += out_softmax[0][1]
                                neu += out_softmax[0][2]

                            else:
                                moc = len(emb) // 50
                                nam = len(emb) % 50
                                for m in range(moc):
                                    inp = emb[m*50: (m+1)*50]
                                    inp_t = [torch.Tensor(e).unsqueeze(0) for e in inp]
                                    inp_cat = torch.cat(inp_t, dim=0).unsqueeze(0).cuda()
                                    out = model(inp_cat)
                                    out_softmax = F.softmax(out).tolist()
                                    pos += out_softmax[0][0]
                                    neg += out_softmax[0][1]
                                    neu += out_softmax[0][2]
                                inp = emb[moc*50: moc*50 + nam]
                                for i in range(50 - len(inp)):
                                    inp.append(word2vec.get_vector('pad'))
                                inp_t = [torch.Tensor(e).unsqueeze(0) for e in inp]
                                inp_cat = torch.cat(inp_t, dim=0).unsqueeze(0).cuda()
                                out = model(inp_cat)
                                out_softmax = F.softmax(out).tolist()
                                pos += out_softmax[0][0]
                                neg += out_softmax[0][1]
                                neu += out_softmax[0][2]

                var_dic['pos'] = str(pos)
                var_dic['neg'] = str(neg)
                var_dic['neu'] = str(neu)
                try:
                    word2vec_list.write(var_dic['CIK'] + '\t' + var_dic['REPORT PERIOD END DATE'] + '\t' + var_dic['FILE DATE'] + '\t' + var_dic['FORM TYPE'] + '\t' + var_dic['pos'] + '\t' + var_dic['neg'] + '\t' + var_dic['neu'] + '\n')
                except:
                    word2vec_list.write('no CIK' + '\n')
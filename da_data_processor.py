import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nltk
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
import os
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True)
args = parser.parse_args()

def entropy(after_softmax):
    #torch.sum dim
    entropy = torch.sum(- after_softmax * torch.log(after_softmax + 1e-10), dim=1, keepdim=True)
    entropy_norm = entropy / np.log(after_softmax.size(1))
    entropy_norm = entropy_norm.squeeze(1)
    entropy_norm = entropy_norm.cpu()
    return entropy_norm

model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert').cuda()
tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')

result_data = open('./data_for_da.txt', 'w')


ld = os.listdir(args.data_dir)
random.shuffle(ld)
count = 0
data_id = 0
for txt in ld:
    if '.txt' in txt and 'download' not in txt and 'DOWNLOADLOG' not in txt and 'newfile' not in txt and 'temp' not in txt and '._' not in txt:
        count += 1
        if count <= 1200:
            fl = open(os.path.join(args.data_dir, txt), errors='ignore')
            fl_read = fl.readlines()
            for fr in fl_read:
                if '<HEADER>'in fr or '</HEADER>' in fr or 'COMPANY NAME' in fr or 'CIK' in fr or 'SIC' in fr or 'FORM TYPE' in fr or 'REPORT PERIOD END DATE' in fr or 'FILE DATE' in fr or '<SECTION>' in fr or '</SECTION>' in fr:
                    pass
                else:
                    lines = nltk.sent_tokenize(fr)
                    for i in lines:
                        tok = tokenizer(i)
                        if len(tok['input_ids']) <= 512:
                            tok['input_ids'] = torch.LongTensor(tok['input_ids']).unsqueeze(0).cuda()
                            tok['token_type_ids'] = torch.LongTensor(tok['token_type_ids']).unsqueeze(0).cuda()
                            tok['attention_mask'] = torch.LongTensor(tok['attention_mask']).unsqueeze(0).cuda()
                            output = model(**tok)
                            logits = output.logits
                            logits = F.softmax(logits)
                            if entropy(logits) <= 0.2:
                                pseudo_label = torch.argmax(logits, dim=1).tolist()[0]
                                result_data.write(str(data_id) + '\t' + i + '\t' + str(pseudo_label) + '\n')
                                data_id += 1
                        else:
                            pass

        else:
            pass

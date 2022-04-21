import os
import nltk
import torch
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig, AutoModelForSequenceClassification
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True)
parser.add_argument('--is_da', required=True)
parser.add_argument('--da_model_dir', required=False)
args = parser.parse_args()

dir = [args.data_dir]
bert_final_list = open('./bert_result.txt', 'w')
bert_final_list.write(
    'CIK' + '\t' + 'REPORT PERIOD END DATE' + '\t' + 'FILE DATE' + '\t' + 'FORM TYPE' + '\t' + 'POS' + '\t' + 'NEG' + '\t' + 'NEU' + '\n')

if args.is_da == True:
    model = AutoModelForSequenceClassification.from_pretrained(args.da_model_dir)
    model.cuda()
    tokenizer = BertTokenizer.from_pretrained('ProsusAI/finBERT')
else:
    model = BertForSequenceClassification.from_pretrained('ProsusAI/finBERT').cuda()
    tokenizer = BertTokenizer.from_pretrained('ProsusAI/finBERT')


def chunking(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]


for d in dir:
    for txt in os.listdir(d):
        if '.txt' in txt and 'download' not in txt and 'DOWNLOADLOG' not in txt and 'newfile' not in txt and 'temp' not in txt and '._' not in txt:
            fl = open(os.path.join(d, txt), 'r', errors='ignore')
            fl_read = fl.readlines()
            var_dic = {}
            pos = 0
            neg = 0
            neu = 0

            for fr in fl_read:

                if '<HEADER>' in fr or '</HEADER>' in fr or 'COMPANY NAME' in fr or 'CIK' in fr or 'SIC' in fr or 'FORM TYPE' in fr or 'REPORT PERIOD END DATE' in fr or 'FILE DATE' in fr or '<SECTION>' in fr or '</SECTION>' in fr:
                    if 'CIK' in fr:
                        var_dic['CIK'] = fr.strip('CIK: ').strip('\n')
                    if 'REPORT PERIOD END DATE' in fr:
                        var_dic['REPORT PERIOD END DATE'] = fr.strip('REPORT PERIOD END DATE: ').strip('\n')
                    if 'FILE DATE' in fr:
                        var_dic['FILE DATE'] = fr.strip('FILE DATE: ').strip('\n')
                    if 'FORM TYPE' in fr:
                        var_dic['FORM TYPE'] = fr.strip('FORM TYPE: ').strip('\n')


                else:
                    if 'CIK' in var_dic.keys() and 'REPORT PERIOD END DATE' in var_dic.keys():
                        if 1 == 1:

                            lines = nltk.sent_tokenize(fr)
                            tok_lines = tokenizer(lines, padding='max_length')

                            tok_15 = {'input_ids': [], 'token_type_ids': [], 'attention_mask': []}
                            tok_30 = {'input_ids': [], 'token_type_ids': [], 'attention_mask': []}
                            tok_50 = {'input_ids': [], 'token_type_ids': [], 'attention_mask': []}
                            tok_100 = {'input_ids': [], 'token_type_ids': [], 'attention_mask': []}
                            tok_512 = {'input_ids': [], 'token_type_ids': [], 'attention_mask': []}

                            chank_1_15 = []
                            chank_1_30 = []
                            chank_1_50 = []
                            chank_1_100 = []
                            chank_1_512 = []
                            chank_2_15 = []
                            chank_2_30 = []
                            chank_2_50 = []
                            chank_2_100 = []
                            chank_2_512 = []
                            chank_3_15 = []
                            chank_3_30 = []
                            chank_3_50 = []
                            chank_3_100 = []
                            chank_3_512 = []

                            for i in tok_lines['input_ids']:
                                if len(i) > 512:
                                    chunk = chunking(i, 512)
                                    for c in chunk[:-1]:
                                        chank_1_512.append(c)
                                    if len(chunk[-1]) <= 15:
                                        for p in range(15 - len(chunk[-1])):
                                            chunk[-1].append(0)
                                        chank_1_15.append(chunk[-1])
                                    elif len(chunk[-1]) <= 30:
                                        for p in range(30 - len(chunk[-1])):
                                            chunk[-1].append(0)
                                        chank_1_30.append(chunk[-1])
                                    elif len(chunk[-1]) <= 50:
                                        for p in range(50 - len(chunk[-1])):
                                            chunk[-1].append(0)
                                        chank_1_50.append(chunk[-1])
                                    elif len(chunk[-1]) <= 100:
                                        for p in range(100 - len(chunk[-1])):
                                            chunk[-1].append(0)
                                        chank_1_100.append(chunk[-1])
                                    else:
                                        for p in range(512 - len(chunk[-1])):
                                            chunk[-1].append(0)
                                        chank_1_512.append(chunk[-1])
                                else:
                                    if len(i) <= 15:
                                        tok_l5['input_ids'].append(i)
                                    elif len(i) <= 30:
                                        tok_30['input_ids'].append(i)
                                    elif len(i) <= 50:
                                        tok_50['input_ids'].append(i)
                                    elif len(i) <= 100:
                                        tok_100['input_ids'].append(i)
                                    else:
                                        tok_512['input_ids'].append(i)

                            for i in tok_lines['token_type_ids']:
                                if len(i) > 512:
                                    chunk = chunking(i, 512)
                                    for c in chunk[:-1]:
                                        chank_2_512.append(c)
                                    if len(chunk[-1]) <= 15:
                                        for p in range(15 - len(chunk[-1])):
                                            chunk[-1].append(0)
                                        chank_2_15.append(chunk[-1])
                                    elif len(chunk[-1]) <= 30:
                                        for p in range(30 - len(chunk[-1])):
                                            chunk[-1].append(0)
                                        chank_2_30.append(chunk[-1])
                                    elif len(chunk[-1]) <= 50:
                                        for p in range(50 - len(chunk[-1])):
                                            chunk[-1].append(0)
                                        chank_2_50.append(chunk[-1])
                                    elif len(chunk[-1]) <= 100:
                                        for p in range(100 - len(chunk[-1])):
                                            chunk[-1].append(0)
                                        chank_2_100.append(chunk[-1])
                                    else:
                                        for p in range(512 - len(chunk[-1])):
                                            chunk[-1].append(0)
                                        chank_2_512.append(chunk[-1])
                                else:
                                    if len(i) <= 15:
                                        tok_l5['token_type_ids'].append(i)
                                    elif len(i) <= 30:
                                        tok_30['token_type_ids'].append(i)
                                    elif len(i) <= 50:
                                        tok_50['token_type_ids'].append(i)
                                    elif len(i) <= 100:
                                        tok_100['token_type_ids'].append(i)
                                    else:
                                        tok_512['token_type_ids'].append(i)

                            for i in tok_lines['attention_mask']:
                                if len(i) > 512:
                                    chunk = chunking(i, 512)
                                    for c in chunk[:-1]:
                                        chank_3_512.append(c)
                                    if len(chunk[-1]) <= 15:
                                        for p in range(15 - len(chunk[-1])):
                                            chunk[-1].append(0)
                                        chank_3_15.append(chunk[-1])
                                    elif len(chunk[-1]) <= 30:
                                        for p in range(30 - len(chunk[-1])):
                                            chunk[-1].append(0)
                                        chank_3_30.append(chunk[-1])
                                    elif len(chunk[-1]) <= 50:
                                        for p in range(50 - len(chunk[-1])):
                                            chunk[-1].append(0)
                                        chank_3_50.append(chunk[-1])
                                    elif len(chunk[-1]) <= 100:
                                        for p in range(100 - len(chunk[-1])):
                                            chunk[-1].append(0)
                                        chank_3_100.append(chunk[-1])
                                    else:
                                        for p in range(512 - len(chunk[-1])):
                                            chunk[-1].append(0)
                                        chank_3_512.append(chunk[-1])
                                else:
                                    if len(i) <= 15:
                                        tok_l5['attention_mask'].append(i)
                                    elif len(i) <= 30:
                                        tok_30['attention_mask'].append(i)
                                    elif len(i) <= 50:
                                        tok_50['attention_mask'].append(i)
                                    elif len(i) <= 100:
                                        tok_100['attention_mask'].append(i)
                                    else:
                                        tok_512['attention_mask'].append(i)

                            tok_15['attention_mask'] = torch.LongTensor(tok_15['attention_mask'] + chank_3_15)
                            tok_15['input_ids'] = torch.LongTensor(tok_15['input_ids'] + chank_1_15)
                            tok_15['token_type_ids'] = torch.LongTensor(tok_15['token_type_ids'] + chank_2_15)
                            tok_30['attention_mask'] = torch.LongTensor(tok_30['attention_mask'] + chank_3_30)
                            tok_30['input_ids'] = torch.LongTensor(tok_30['input_ids'] + chank_1_30)
                            tok_30['token_type_ids'] = torch.LongTensor(tok_30['token_type_ids'] + chank_2_30)
                            tok_50['attention_mask'] = torch.LongTensor(tok_50['attention_mask'] + chank_3_50)
                            tok_50['input_ids'] = torch.LongTensor(tok_50['input_ids'] + chank_1_50)
                            tok_50['token_type_ids'] = torch.LongTensor(tok_50['token_type_ids'] + chank_2_50)
                            tok_100['attention_mask'] = torch.LongTensor(tok_100['attention_mask'] + chank_3_100)
                            tok_100['input_ids'] = torch.LongTensor(tok_100['input_ids'] + chank_1_100)
                            tok_100['token_type_ids'] = torch.LongTensor(tok_100['token_type_ids'] + chank_2_100)
                            tok_512['attention_mask'] = torch.LongTensor(tok_512['attention_mask'] + chank_3_512)
                            tok_512['input_ids'] = torch.LongTensor(tok_512['input_ids'] + chank_1_512)
                            tok_512['token_type_ids'] = torch.LongTensor(tok_512['token_type_ids'] + chank_2_512)

                            toks = [tok_15, tok_30, tok_50, tok_100, tok_512]
                            tokss = []
                            for t in toks:
                                if not t['attention_mask'].shape[0] == 0:
                                    tokss.append(t)

                            for t in tokss:

                                if t['input_ids'] != []:
                                    if t['input_ids'].shape[0] <= 16:
                                        t['input_ids'] = t['input_ids'].cuda()
                                        t['token_type_ids'] = t['token_type_ids'].cuda()
                                        t['attention_mask'] = t['attention_mask'].cuda()

                                        output = model(**t)
                                        logits = output.logits
                                        logits = F.softmax(logits, dim=-1)
                                        sums = torch.sum(logits, dim=0).tolist()

                                        pos += sums[0]
                                        neg += sums[1]
                                        neu += sums[2]
                                    else:
                                        pos, neg, neu = 0, 0, 0
                                        nem = t['input_ids'].shape[0] // 16
                                        mec = t['input_ids'].shape[0] % 16
                                        for i in range(nem):
                                            tak_batch = {}
                                            tak_batch['input_ids'] = t['input_ids'][i * 16: (i + 1) * 16].cuda()
                                            tak_batch['token_type_ids'] = t['token_type_ids'][
                                                                          i * 16: (i + 1) * 16].cuda()
                                            tak_batch['attention_mask'] = t['attention_mask'][
                                                                          i * 16: (i + 1) * 16].cuda()
                                            output = model(**tak_batch)
                                            logits = output.logits
                                            logits = F.softmax(logits, dim=-1)
                                            sums = torch.sum(logits, dim=0).tolist()
                                            pos += sums[0]
                                            neg += sums[1]
                                            neu += sums[2]
                                        tak_batch = {}
                                        tak_batch['input_ids'] = t['input_ids'][nem * 16: nem * 16 + mec].cuda()
                                        tak_batch['token_type_ids'] = t['token_type_ids'][
                                                                      nem * 16: nem * 16 + mec].cuda()
                                        tak_batch['attention_mask'] = t['attention_mask'][
                                                                      nem * 16: nem * 16 + mec].cuda()
                                        output = model(**tak_batch)
                                        logits = output.logits
                                        logits = F.softmax(logits, dim=-1)
                                        sums = torch.sum(logits, dim=0).tolist()
                                        pos += sums[0]
                                        neg += sums[1]
                                        neu += sums[2]

            var_dic['pos'] = str(pos)
            var_dic['neg'] = str(neg)
            var_dic['neu'] = str(neu)
            try:
                bert_final_list.write(
                    var_dic['CIK'] + '\t' + var_dic['REPORT PERIOD END DATE'] + '\t' + var_dic['FILE DATE'] + '\t' +
                    var_dic['FORM TYPE'] + '\t' + var_dic['pos'] + '\t' + var_dic['neg'] + '\t' + var_dic['neu'] + '\n')
            except:
                bert_final_list.write('no CIK' + '\n')

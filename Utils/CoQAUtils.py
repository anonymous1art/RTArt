# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re
import os
import sys
import random
import string
import logging
import argparse
import unicodedata
from shutil import copyfile
from datetime import datetime
from collections import Counter
from collections import defaultdict
import torch
import msgpack
import json
import numpy as np
import pandas as pd
from Models.Bert.tokenization import BertTokenizer
from Utils.GeneralUtils import normalize_text, nlp
from Utils.Constants import *
from torch.autograd import Variable

POS = {w: i for i, w in enumerate([''] + list(nlp.tagger.labels))}
ENT = {w: i for i, w in enumerate([''] + nlp.entity.move_names)}

def build_embedding(embed_file, targ_vocab, wv_dim):
    vocab_size = len(targ_vocab)
    emb = np.random.uniform(-1, 1, (vocab_size, wv_dim))
    emb[0] = 0 # <PAD> should be all 0 (using broadcast)

    w2id = {w: i for i, w in enumerate(targ_vocab)}
    lineCnt = 0
    with open(embed_file, encoding="utf8") as f:
        for line in f:
            lineCnt = lineCnt + 1
            if lineCnt % 100000 == 0:
                print('.', end = '',flush=True)
            elems = line.split()
            token = normalize_text(''.join(elems[0:-wv_dim]))
            if token in w2id:
                emb[w2id[token]] = [float(v) for v in elems[-wv_dim:]]
    return emb

def token2id_sent(sent, w2id, unk_id=None, to_lower=False, takenize=False):
    if to_lower:
        sent = sent.lower()
    w2id_len = len(w2id)
    if takenize:
        ids = []
        m_1 = m_2 = 0
        for w in sent:
            if w in w2id:
                ids.append(w2id[w])
            else:
                m_1 += 1
                w_l = len(w)
                flag = False
                for l in [w_l-1, w_l-2]:
                    flag = False
                    for i in range(w_l):
                        if i + l > w_l:
                            break
                        new_w = w[i:i+l]
                        if new_w in w2id:
                            m_2 += 1
                            ids.append(w2id[new_w])
                            flag = True
                            break
                    if flag:
                        break
                if flag == False:
                    ids.append(unk_id)
    else:    
        ids = [w2id[w] if w in w2id else unk_id for w in sent]
    oov = sum([1 if item == unk_id else 0 for item in ids])
    # oov_per = oov / len(ids)
    if takenize:
        return ids, oov, len(ids), m_1, m_2
    else:
        return ids, oov, len(ids)

def char2id_sent(sent, c2id, unk_id=None, to_lower=False):
    if to_lower:
        sent = sent.lower()
    cids = [[c2id["<STA>"]] + [c2id[c] if c in c2id else unk_id for c in w] + [c2id["<END>"]] for w in sent]
    return cids

def token2id(w, vocab, unk_id=None):
    return vocab[w] if w in vocab else unk_id

'''
 Generate feature per context word according to its exact match with question words
'''
def feature_gen(context, question):
    counter_ = Counter(w.text.lower() for w in context)
    total = sum(counter_.values())
    term_freq = [counter_[w.text.lower()] / total for w in context]
    question_word = {w.text for w in question}
    question_lower = {w.text.lower() for w in question}
    question_lemma = {w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower() for w in question}
    match_origin = [w.text in question_word for w in context]
    match_lower = [w.text.lower() in question_lower for w in context]
    match_lemma = [(w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower()) in question_lemma for w in context]
    C_features = list(zip(term_freq, match_origin, match_lower, match_lemma))
    return C_features    

'''
 Get upper triangle matrix from start and end scores (batch)
 Input:
  score_s: batch x context_len
  score_e: batch x context_len
  context_len: number of words in context
  max_len: maximum span of answer
  use_cuda: whether GPU is used
 Output:
  expand_score: batch x (context_len * context_len) 
'''
def gen_upper_triangle(score_s, score_e, max_len, use_cuda):
    batch_size = score_s.shape[0]
    context_len = score_s.shape[1]
    # batch x context_len x context_len
    expand_score = score_s.unsqueeze(2).expand([batch_size, context_len, context_len]) +\
        score_e.unsqueeze(1).expand([batch_size, context_len, context_len])
    score_mask = torch.ones(context_len)
    if use_cuda:
        score_mask = score_mask.cuda()
    score_mask = torch.ger(score_mask, score_mask).triu().tril(max_len - 1)
    empty_mask = score_mask.eq(0).unsqueeze(0).expand_as(expand_score)
    expand_score.data.masked_fill_(empty_mask.data, -float('inf'))
    return expand_score.contiguous().view(batch_size, -1) # batch x (context_len * context_len)    

class BatchGen:
    def __init__(self, opt, data, use_cuda, vocab, char_vocab, train_img_id2idx, train_img_features, train_img_spatials, val_img_id2idx, val_img_features, val_img_spatials, mod='train'):
        # file_name = os.path.join(self.spacyDir, 'coqa-' + dataset_label + '-preprocessed.json')

        self.data = data
        self.use_cuda = use_cuda 
        self.vocab = vocab
        self.char_vocab = char_vocab
        self.train_img_features = train_img_features
        self.train_img_id2idx = train_img_id2idx
        self.train_img_spatials = train_img_spatials
        self.val_img_features = val_img_features
        self.val_img_spatials = val_img_spatials
        self.val_img_id2idx = val_img_id2idx
        self.img_feature_dim = opt['img_fea_dim']
        self.img_spatial_dim = opt['img_spa_dim']
        self.img_fea_num = opt['img_fea_num']
        self.use_img_feature = 'img_feature' in opt

        # self.dis = 'dis' in opt['selective_source']
        # self.ocr = 'ocr' in opt['selective_source']
        # self.n_gram = '2-gram' in opt['selective_source']
        # self.ensemble = 'ensemble' in opt['selective_source']
        self.ocr_name_list = opt['ocr_name_list'].split('#')
        error_ocr_name = []
        for ocr_name in self.ocr_name_list:
            if ocr_name not in self.data[0]:
                error_ocr_name.append(ocr_name)
        if len(error_ocr_name) != 0:
            print('OCR name ERROR: ', error_ocr_name)
            assert False
        else:
            print('Using OCR from: ', self.ocr_name_list)
        if mod == 'train':
            self.evaluation = False
        else:
            self.evaluation = True
        self.opt = opt
        if 'PREV_ANS' in self.opt:
            self.prev_ans = self.opt['PREV_ANS']
        else:
            self.prev_ans = 2

        if 'PREV_QUES' in self.opt:
            self.prev_ques = self.opt['PREV_QUES']
        else:
            self.prev_ques = 0

        self.use_char_cnn = 'CHAR_CNN' in self.opt

        self.bert_tokenizer = None
        if 'BERT' in self.opt:
            if 'BERT_LARGE' in opt:
                print('Using BERT Large model')
                tokenizer_file = os.path.join(opt['datadir'], opt['BERT_large_tokenizer_file'])
                print('Loading tokenizer from', tokenizer_file)
                self.bert_tokenizer = BertTokenizer.from_pretrained(tokenizer_file)
            else:
                # print('Using BERT base model')
                tokenizer_file = os.path.join(opt['datadir'], opt['BERT_tokenizer_file'])
                # print('Loading tokenizer from', tokenizer_file)
                self.bert_tokenizer = BertTokenizer.from_pretrained(tokenizer_file)

        self.answer_span_in_context = 'ANSWER_SPAN_IN_CONTEXT_FEATURE' in self.opt

        self.ques_max_len = (30 + 1) * self.prev_ans + (25 + 1) * (self.prev_ques + 1)
        self.char_max_len = 30

        # print('*****************')
        # print('prev_ques   :', self.prev_ques)
        # print('prev_ans    :', self.prev_ans)
        # print('ques_max_len:', self.ques_max_len)
        # print('*****************')

        c2id = {c: i for i, c in enumerate(char_vocab)}
        
        # random shuffle for training
        # indices = list(range(len(self.data)))
        # random.shuffle(indices)
        # self.data = [self.data[i] for i in indices]
        if mod == 'dev':
            self.data = self.data[-self.opt['dev_num']:]
        elif mod == 'train':
            self.data = self.data[:-self.opt['dev_num']]
        elif mod == 'test':
            self.data = self.data
        else:
            print('mod ERROR: {}'.format(mod))
            assert False
        self.od_name_list = opt['od_name_list'].split('#')
        error_od_name = []
        for od_name in self.od_name_list:
            if od_name not in self.data[0]:
                error_od_name.append(od_name)
        if len(error_od_name) != 0:
            print('OD name ERROR: ', error_ocr_name)
            assert False
        else:
            print('Using OD from: ', self.od_name_list)
        # if opt['yoloV3_9000'] == 'V3':
        #     print('yoloV3_9000: V3')
        # elif opt['yoloV3_9000'] == '_9000':
        #     print('yoloV3_9000: 9000')
        #     # for datum in self.data:
        #     #     datum['OD'] = datum['YOLO']
        # elif opt['yoloV3_9000'] == 'V3_9000':
        #     print('yoloV3_9000: V3_9000')
        #     # for datum in self.data:
        #     #     for _yolo in datum['YOLO']:
        #     #         datum['OD'].append(_yolo)
        # else:
        #     print('yoloV3_9000 parameter is incorrect!')
        #     assert False

    def __len__(self):
        return (len(self.data) + self.opt['batch_size'] - 1) // self.opt['batch_size']

    def bertify(self, words):
        if self.bert_tokenizer is None:
            return None

        bpe = ['[CLS]']
        x_bert_offsets = []
        for word in words:
            now = self.bert_tokenizer.tokenize(word)
            x_bert_offsets.append([len(bpe), len(bpe) + len(now)])
            bpe.extend(now)
        if len(words) == 0:
            x_bert_offsets = [1,1]
        
        bpe.append('[SEP]')

        x_bert = self.bert_tokenizer.convert_tokens_to_ids(bpe)
        return x_bert, x_bert_offsets
    def My_bertify(self, sentence):
        if self.bert_tokenizer is None:
            return None

        bpe = ['[CLS]']
        now = self.bert_tokenizer.tokenize(sentence)
        bpe.extend(now)
        
        bpe.append('[SEP]')

        x_bert = self.bert_tokenizer.convert_tokens_to_ids(bpe)
        return x_bert

    def __iter__(self):
        data = self.data
        MAX_ANS_SPAN = 15
        data_idx = range(0, len(data))
        batch_size_t = self.opt['batch_size']
        if not self.evaluation:
            data_idx = np.random.permutation(data_idx)
        
        if ('GLOVE' not in self.opt) and ('BERT' in self.opt):
            for batch_i in range((len(data_idx) + batch_size_t - 1) // batch_size_t):
                batch_idx = data_idx[batch_size_t*batch_i:batch_size_t*(batch_i+1)]
                batch_size = len(batch_idx)
                batches = [data[item] for item in batch_idx]
                q_list = []
                od_list = []
                ocr_list = []
                file_name = []
                answers = []
                od_offset_list = []
                ocr_offset_list = []
                od_max_num = ocr_max_num = 0
                ocr_ans_list = []
                pad_pos = [0.0 for i in range(8)]
                for datum in batches:
                    tmp = []
                    datum['ocr_len'] = len(datum['OCR'])
                    if self.dis:
                        tmp += datum['distractors']
                    if self.ocr:
                        tmp += datum['OCR']
                    if self.n_gram:
                        tmp += datum['2_grams']
                    datum['OCR'] = tmp
                    q_list.append(self.My_bertify(datum['question'] + '<Q>'))
                    od_st = len(od_list)
                    _od_max_num = _ocr_max_num = 1
                    for _od in datum['OD']:
                        od_list.append(self.My_bertify(_od['original']))
                        _od_max_num += 1
                    od_list.append(self.My_bertify('<OD>'))
                    od_offset_list.append([od_st, len(od_list)])
                    ocr_st = len(ocr_list)
                    _ocr_ans_list = []
                    for _ocr in datum['OCR']:
                        ocr_list.append(self.My_bertify(_ocr['original']))
                        _ocr_ans_list.append(_ocr['original'])
                        _ocr_max_num += 1
                    ocr_list.append(self.My_bertify('<OCR>'))
                    ocr_ans_list.append(_ocr_ans_list)
                    ocr_offset_list.append([ocr_st, len(ocr_list)])
                    file_name.append(datum['filename'])
                    answers.append(datum['orign_answers'])
                    od_max_num = max(od_max_num, _od_max_num)
                    ocr_max_num = max(ocr_max_num, _ocr_max_num)
                q_num = len(q_list)
                od_num = len(od_list)
                ocr_num = len(ocr_list)
                q_len = max([len(item) for item in q_list])
                od_len = max([len(item) for item in od_list])
                ocr_len = max([len(item) for item in ocr_list])

                q_bert = torch.LongTensor(q_num, q_len).fill_(0)
                od_bert = torch.LongTensor(od_num, od_len).fill_(0)
                od_position = torch.FloatTensor(batch_size, od_max_num, 8).fill_(0)
                ocr_bert = torch.LongTensor(ocr_num, ocr_len).fill_(0)
                ocr_position = torch.FloatTensor(batch_size, ocr_max_num, 8).fill_(0)
                od_last_index = []
                ocr_last_index = []
                for _i, _item in enumerate(q_list):
                    selected_len = len(_item)
                    q_bert[_i][:selected_len] = torch.LongTensor(_item)
                for _i, _item in enumerate(od_list):
                    selected_len = len(_item)
                    od_bert[_i][:selected_len] = torch.LongTensor(_item)
                    od_last_index.append(selected_len - 2)
                for _i, _item in enumerate(ocr_list):
                    selected_len = len(_item)
                    ocr_bert[_i][:selected_len] = torch.LongTensor(_item)
                    ocr_last_index.append(selected_len - 2)
                for datum_i, datum in enumerate(batches):
                    for _i, _ocr in enumerate(datum['OCR']):
                        ocr_position[datum_i][_i] = torch.LongTensor(_ocr['pos'])
                    for _i, _od in enumerate(datum['OD']):
                        od_position[datum_i][_i] = torch.LongTensor(_od['pos'])
                q_mask = 1 - q_bert.eq(0)
                od_mask = 1 - od_bert.eq(0)
                ocr_mask = 1 - ocr_bert.eq(0)


                ground_truth_YN = torch.LongTensor(batch_size).fill_(0)
                ground_truth = torch.LongTensor(batch_size, ocr_max_num).fill_(0)
                for datum_i, datum in enumerate(batches):
                    selected_len = len(datum['answers_id_ocr'])
                    if selected_len == 0:
                        ground_truth_YN[datum_i] = 1
                    else:
                        ground_truth_YN[datum_i] = 0
                        for item in datum['answers_id_ocr']:
                            ground_truth[datum_i][item] = 1



                
                if self.use_cuda:
                    q_bert = Variable(q_bert.cuda(async=True))
                    q_mask = Variable(q_mask.cuda(async=True))
                    od_bert = Variable(od_bert.cuda(async=True))
                    od_mask = Variable(od_mask.cuda(async=True))
                    od_position = Variable(od_position.cuda(async=True))
                    ocr_bert = Variable(ocr_bert.cuda(async=True))
                    ocr_mask = Variable(ocr_mask.cuda(async=True))
                    ocr_position = Variable(ocr_position.cuda(async=True))
                    ground_truth_YN = Variable(ground_truth_YN.cuda(async=True))
                    ground_truth = Variable(ground_truth.cuda(async=True))
                else:
                    q_bert = Variable(q_bert)
                    q_mask = Variable(q_mask)
                    od_bert = Variable(od_bert)
                    od_mask = Variable(od_mask)
                    od_position = Variable(od_position)
                    ocr_bert = Variable(ocr_bert)
                    ocr_mask = Variable(ocr_mask)
                    ocr_position = Variable(ocr_position)
                    ground_truth_YN = Variable(ground_truth_YN)
                    ground_truth = Variable(ground_truth)
                yield(None, None, None, None, None, od_position, None, od_bert, od_mask, None, None, q_mask, None, None, None, None, q_bert, q_mask, None, None, None, None, None, None, ocr_position, None, ocr_bert, ocr_mask, None, od_offset_list, ocr_offset_list, od_last_index, ocr_last_index, od_max_num, ocr_max_num,ground_truth, ground_truth_YN, ocr_ans_list, file_name, answers)
        else:
            for batch_i in range((len(data_idx) + batch_size_t - 1) // batch_size_t):
            # for batch_i in range(4):
                batch_idx = data_idx[batch_size_t*batch_i:batch_size_t*(batch_i+1)]
                batch_size = len(batch_idx)
                batches = [data[item] for item in batch_idx]
                # while sum([len(t['OCR']) for t in batches]) > 700:


                que_len = max([len(datum['annotated_question']['wordid']) for datum in batches])
                q = torch.LongTensor(batch_size, que_len+1).fill_(0)
                q_char = torch.LongTensor(batch_size, que_len+1,self.char_max_len).fill_(0)
                q_pos = torch.LongTensor(batch_size, que_len+1).fill_(0)
                q_ent = torch.LongTensor(batch_size, que_len+1).fill_(0)
                img_fea = torch.FloatTensor(batch_size, self.img_fea_num, self.img_feature_dim).fill_(0)
                img_spa = torch.FloatTensor(batch_size, self.img_fea_num, 8).fill_(0)
                file_name = []
                answers = []
                ocr_num = od_num = 0
                ocr_max_len = od_max_len = 0
                ocr_max_num = od_max_num = 0
                ocr_ans_list = []
                question_id = []
                original_ocr = []
                original_od = []
                for datum_i, datum in enumerate(batches):
                    q_id = datum['question_id']
                    question_id.append(q_id)
                    if self.use_img_feature:
                        if q_id in self.train_img_id2idx.keys():
                            q_idx = self.train_img_id2idx[q_id]
                            img_fea[datum_i] = self.train_img_features[q_idx]
                            img_spa[datum_i,:,0] = self.train_img_spatials[q_idx,:,0]
                            img_spa[datum_i,:,1] = self.train_img_spatials[q_idx,:,1]
                            img_spa[datum_i,:,2] = self.train_img_spatials[q_idx,:,2]
                            img_spa[datum_i,:,3] = self.train_img_spatials[q_idx,:,1]
                            img_spa[datum_i,:,4] = self.train_img_spatials[q_idx,:,2]
                            img_spa[datum_i,:,5] = self.train_img_spatials[q_idx,:,3]
                            img_spa[datum_i,:,6] = self.train_img_spatials[q_idx,:,0]
                            img_spa[datum_i,:,7] = self.train_img_spatials[q_idx,:,3]
                        else:
                            q_idx = self.val_img_id2idx[q_id]
                            img_fea[datum_i] = self.val_img_features[q_idx]
                            img_spa[datum_i,:,0] = self.val_img_spatials[q_idx,:,0]
                            img_spa[datum_i,:,1] = self.val_img_spatials[q_idx,:,1]
                            img_spa[datum_i,:,2] = self.val_img_spatials[q_idx,:,2]
                            img_spa[datum_i,:,3] = self.val_img_spatials[q_idx,:,1]
                            img_spa[datum_i,:,4] = self.val_img_spatials[q_idx,:,2]
                            img_spa[datum_i,:,5] = self.val_img_spatials[q_idx,:,3]
                            img_spa[datum_i,:,6] = self.val_img_spatials[q_idx,:,0]
                            img_spa[datum_i,:,7] = self.val_img_spatials[q_idx,:,3]
                    tmp = []
                    datum['ocr_len'] = len(datum['OCR'])
                    original_ocr.append(datum['OCR'])
                    for _name in self.ocr_name_list:
                        tmp += datum[_name]
                    # if self.dis:
                    #     tmp += datum['distractors']
                    # if self.ocr:
                    #     tmp += datum['OCR']
                    # if self.n_gram:
                    #     tmp += datum['2_grams']
                    # if self.ensemble:
                    #     tmp += datum['ensemble_ocr']
                    datum['OCR'] = tmp


                    original_od.append(datum['OD'])
                    tmp = []
                    for _name in self.od_name_list:
                        tmp += datum[_name]
                    # if self.opt['yoloV3_9000'] == 'V3':
                    #     tmp += datum['OD']
                    # elif self.opt['yoloV3_9000'] == '_9000':
                    #     # print('yoloV3_9000: 9000')
                    #     # for datum in self.data:
                    #     #     datum['OD'] = datum['YOLO']
                    #     tmp += datum['YOLO']
                    # elif self.opt['yoloV3_9000'] == 'V3_9000':
                    #     # print('yoloV3_9000: V3_9000')
                    #     # for datum in self.data:
                    #     #     for _yolo in datum['YOLO']:
                    #     #         datum['OD'].append(_yolo)
                    #     tmp += datum['OD']
                    #     tmp += datum['YOLO']
                    datum['OD'] = tmp
                    selected_len = len(datum['annotated_question']['wordid'])
                    q[datum_i][:selected_len] = torch.LongTensor(datum['annotated_question']['wordid'])
                    q[datum_i][selected_len] = torch.LongTensor([2])
                    q_pos[datum_i][:selected_len] = torch.LongTensor(datum['annotated_question']['pos_id'])
                    q_ent[datum_i][:selected_len] = torch.LongTensor(datum['annotated_question']['ent_id'])
                    _ocr_ans_list = []
                    _od_max_len = _ocr_max_len = 0
                    for _ocr in datum['OCR']:
                        _ocr_ans_list.append(_ocr['original'])
                        _ocr_max_len = max(_ocr_max_len, len(_ocr['word']['word']))
                    if len(datum['OD']) != 0:
                        _od_max_len = max([len(_od['object']['word']) for _od in datum['OD']])
                    od_max_len = max(_od_max_len, od_max_len)
                    ocr_max_len = max(_ocr_max_len, ocr_max_len)
                    od_max_num = max(od_max_num, len(datum['OD'])+1)
                    ocr_max_num = max(ocr_max_num, len(datum['OCR'])+1)
                    ocr_ans_list.append(_ocr_ans_list)
                    file_name.append(datum['filename'])
                    answers.append(datum['orign_answers'])
                    ocr_num += len(datum['OCR'])
                    od_num += len(datum['OD'])
                
                if 'BERT' in self.opt:
                    q_bert_offsets = torch.LongTensor(batch_size, que_len+1, 2).fill_(0)
                    x_bert_list = []
                    max_bert_len = 0
                    for datum_i, datum in enumerate(batches):
                        x_bert, x_bert_offsets = self.bertify(datum['annotated_question']['word']+['<Q>'])
                        max_bert_len = max(max_bert_len, len(x_bert))
                        x_bert_list.append(x_bert)
                        # selected_len = len(x_bert)
                        # q_bert[datum_i][:len(x_bert)] = torch.tensor(x_bert, dtype = torch.long)
                        selected_len = len(x_bert_offsets)
                        q_bert_offsets[datum_i][:selected_len] = torch.tensor(x_bert_offsets, dtype = torch.long)
                    q_bert = torch.LongTensor(batch_size, max_bert_len).fill_(0)
                    for idx_i, _bert in enumerate(x_bert_list):
                        selected_len = len(_bert)
                        q_bert[idx_i][:selected_len] = torch.tensor(_bert, dtype = torch.long)
                # max_ocr_len = 0
                # max_od_len = 0
                # for datum in batches:
                #     ocr_len = 0
                #     od_len = 0
                #     for ocr in datum['OCR']:
                #         ocr_len += len(ocr['word']['word'])
                #     for od in datum['OD']:
                #         od_len += len(od['object']['word'])
                #     max_ocr_len = max(ocr_len, max_ocr_len)
                #     max_od_len = max(od_len, max_od_len)
                ocr_max_len = max(1, ocr_max_len)
                od_max_len = max(1, od_max_len)
                ocr = torch.LongTensor(ocr_num+batch_size, ocr_max_len).fill_(0)
                ocr_ent = torch.LongTensor(ocr_num+batch_size, ocr_max_len).fill_(0)
                ocr_pos = torch.LongTensor(ocr_num+batch_size, ocr_max_len).fill_(0)
                ocr_position = torch.FloatTensor(batch_size, ocr_max_num, 8).fill_(0)
                ocr_offset = []
                od = torch.LongTensor(od_num+batch_size, od_max_len).fill_(0)
                od_ent = torch.LongTensor(od_num+batch_size, od_max_len).fill_(0)
                od_pos = torch.LongTensor(od_num+batch_size, od_max_len).fill_(0)
                od_position = torch.FloatTensor(batch_size, od_max_num,8).fill_(0)
                od_offset = []
                ocr_cnt = 0
                od_cnt = 0
                ocr_last_index = []
                od_last_index = []
                for datum_i, datum in enumerate(batches):
                    st = ocr_cnt
                    for _i, _ocr in enumerate(datum['OCR']):
                        selected_len = len(_ocr['word']['wordid'])
                        ocr[ocr_cnt][:selected_len] = torch.LongTensor(_ocr['word']['wordid'])
                        ocr_pos[ocr_cnt][:selected_len] = torch.LongTensor(_ocr['word']['pos_id'])
                        ocr_ent[ocr_cnt][:selected_len] = torch.LongTensor(_ocr['word']['ent_id'])
                        ocr_position[datum_i][_i] = torch.FloatTensor(_ocr['pos'])
                        ocr_cnt += 1
                        ocr_last_index.append(selected_len-1)
                    ocr[ocr_cnt][0] = 3
                    ocr_last_index.append(0)
                    ocr_cnt += 1
                    ocr_offset.append([st, ocr_cnt])
                    st = od_cnt
                    for _i, _od in enumerate(datum['OD']):
                        selected_len = len(_od['object']['wordid'])
                        od[od_cnt][:selected_len] = torch.LongTensor(_od['object']['wordid'])
                        od_pos[od_cnt][:selected_len] = torch.LongTensor(_od['object']['pos_id'])
                        od_ent[od_cnt][:selected_len] = torch.LongTensor(_od['object']['ent_id'])
                        od_position[datum_i][_i] = torch.FloatTensor(_od['pos'])
                        od_cnt += 1
                        od_last_index.append(selected_len-1)
                    od[od_cnt][0] = 4
                    od_last_index.append(0)
                    od_cnt += 1
                    od_offset.append([st, od_cnt])
                if 'BERT' in self.opt:
                    ocr_bert_list = []
                    od_bert_list = []
                    max_ocr_bert_len = max_od_bert_len = 0
                    # ocr_bert = torch.LongTensor(batch_size, max_ocr_len+3).fill_(0)
                    # od_bert = torch.LongTensor(batch_size, max_od_len+3).fill_(0)
                    ocr_bert_offsets = torch.LongTensor(ocr_num+batch_size, ocr_max_len, 2).fill_(0)
                    od_bert_offsets = torch.LongTensor(od_num+batch_size, od_max_len, 2).fill_(0)
                    ocr_cnt = 0
                    od_cnt = 0
                    for datum_i, datum in enumerate(batches):
                        for _ocr in datum['OCR']:
                            x_bert, x_bert_offsets = self.bertify(_ocr['word']['word'])
                            ocr_bert_list.append(x_bert)
                            max_ocr_bert_len = max(max_ocr_bert_len, len(x_bert))
                            selected_len = len(x_bert_offsets)
                            ocr_bert_offsets[ocr_cnt][:selected_len] = torch.LongTensor(x_bert_offsets)
                            ocr_cnt += 1
                        x_bert, x_bert_offsets = self.bertify(['<OCR>'])
                        ocr_bert_list.append(x_bert)
                        max_ocr_bert_len = max(max_ocr_bert_len, len(x_bert))
                        selected_len = len(x_bert_offsets)
                        ocr_bert_offsets[ocr_cnt][:selected_len] = torch.LongTensor(x_bert_offsets)
                        ocr_cnt += 1
                        for _od in datum['OD']:
                            x_bert, x_bert_offsets = self.bertify(_od['object']['word'])
                            od_bert_list.append(x_bert)
                            max_od_bert_len = max(max_od_bert_len, len(x_bert))
                            selected_len = len(x_bert_offsets)
                            od_bert_offsets[od_cnt][:selected_len] = torch.LongTensor(x_bert_offsets)
                            od_cnt += 1
                        x_bert, x_bert_offsets = self.bertify(['<OD>'])
                        od_bert_list.append(x_bert)
                        max_od_bert_len = max(max_od_bert_len, len(x_bert))
                        selected_len = len(x_bert_offsets)
                        od_bert_offsets[od_cnt][:selected_len] = torch.LongTensor(x_bert_offsets)
                        od_cnt += 1
                    ocr_bert = torch.LongTensor(ocr_num+batch_size, max_ocr_bert_len).fill_(0)
                    od_bert = torch.LongTensor(od_num+batch_size, max_od_bert_len).fill_(0)
                    for _idx, _bert in enumerate(ocr_bert_list):
                        selected_len = len(_bert)
                        ocr_bert[_idx][:selected_len] = torch.LongTensor(_bert)
                    for _idx, _bert in enumerate(od_bert_list):
                        selected_len = len(_bert)
                        od_bert[_idx][:selected_len] = torch.LongTensor(_bert)
                for datum_i, datum in enumerate(batches):
                    datum['OCR'] = original_ocr[datum_i]
                for datum_i, datum in enumerate(batches):
                    datum['OD'] = original_od[datum_i]
                q_mask = 1 - q.eq(0)
                ocr_mask = 1 - ocr.eq(0)
                od_mask = 1 - od.eq(0)

                ground_truth_YN = torch.LongTensor(batch_size).fill_(0)
                ground_truth = torch.LongTensor(batch_size, ocr_max_num).fill_(0)
                for datum_i, datum in enumerate(batches):
                    tmp = []
                    for _name in self.ocr_name_list:
                        if _name == 'OCR':
                            tmp.append(datum['answers_id_'+_name]+[datum['ocr_len']])
                        else:
                            tmp.append(datum['answers_id_'+_name]+[len(datum[_name])])
                    # if self.dis:
                    #     tmp.append(datum['answers_id_dis'] + [len(datum['distractors'])])
                    # if self.ocr:
                    #     tmp.append(datum['answers_id_ocr'] + [datum['ocr_len']])
                    # if self.n_gram:
                    #     tmp.append(datum['answers_ocr_gram2'] + [len(datum['2_grams'])])
                    # if self.ensemble:
                    #     tmp.append(datum['answers_id_ensemble'] + [len(datum['ensemble_ocr'])])
                    s_max_idx = s_max = -1
                    for _idx, item in enumerate(tmp):
                        if item[1] > s_max:
                            s_max_idx = _idx
                            s_max = item[1]
                    if s_max < self.opt['ans_threshold']:
                        ground_truth_YN[datum_i] = 1
                    else:
                        answer_idx = 0
                        for i in range(s_max_idx):
                            answer_idx += tmp[i][2]
                        answer_idx += tmp[s_max_idx][0]
                        ground_truth[datum_i][answer_idx] = 1

                    # selected_len = len(datum['answers_id_ocr']) + len(datum['answers_id_dis'])
                    # if selected_len == 0:
                    #     ground_truth_YN[datum_i] = 1
                    #     #assert False
                    # else:
                    #     ground_truth_YN[datum_i] = 0
                    #     for item in datum['answers_id_dis']:
                    #         ground_truth[datum_i][item] = 1
                    #     if not self.only_dis:
                    #         for item in datum['answers_id_ocr']:
                    #             ground_truth[datum_i][item+100] = 1
                if 'BERT' in self.opt:
                    q_bert_mask = 1 - q_bert.eq(0)
                    ocr_bert_mask = 1 - ocr_bert.eq(0)
                    od_bert_mask = 1 - od_bert.eq(0)
                    if self.use_cuda:
                        q_bert = Variable(q_bert.cuda(async=True))
                        q_bert_mask = Variable(q_bert_mask.cuda(async=True))
                        q_bert_offsets = Variable(q_bert_offsets.cuda(async=True))
                        ocr_bert = Variable(ocr_bert.cuda(async=True))
                        ocr_bert_mask = Variable(ocr_bert_mask.cuda(async=True))
                        ocr_bert_offsets = Variable(ocr_bert_offsets.cuda(async=True))
                        od_bert = Variable(od_bert.cuda(async=True))
                        od_bert_mask = Variable(od_bert_mask.cuda(async=True))
                        od_bert_offsets = Variable(od_bert_offsets.cuda(async=True))
                    else:
                        q_bert = Variable(q_bert)
                        q_bert_mask = Variable(q_bert_mask)
                        q_bert_offsets = Variable(q_bert_offsets)
                        ocr_bert = Variable(ocr_bert)
                        ocr_bert_mask = Variable(ocr_bert_mask)
                        ocr_bert_offsets = Variable(ocr_bert_offsets)
                        od_bert = Variable(od_bert)
                        od_bert_mask = Variable(od_bert_mask)
                        od_bert_offsets = Variable(od_bert_offsets)
                else:
                    q_bert = None
                    q_bert_mask = None
                    q_bert_offsets = None
                    ocr_bert = None
                    ocr_bert_mask = None
                    ocr_bert_offsets = None
                    od_bert = None
                    od_bert_mask = None
                    od_bert_offsets = None

                if self.use_char_cnn:
                    _ = 1 
                else:
                    q_char = None
                    q_char_mask = None
                    ocr_char = None
                    ocr_char_mask = None      
                    od_char = None
                    od_char_mask = None                       
                if self.use_cuda:
                    img_fea = Variable(img_fea.cuda(async=True))
                    img_spa = Variable(img_spa.cuda(async=True))
                    q = Variable(q.cuda(async=True))
                    q_mask = Variable(q_mask.cuda(async=True))
                    q_pos = Variable(q_pos.cuda(async=True))
                    q_ent = Variable(q_ent.cuda(async=True))
                    ocr = Variable(ocr.cuda(async=True))
                    ocr_mask = Variable(ocr_mask.cuda(async=True))
                    ocr_pos = Variable(ocr_pos.cuda(async=True))
                    ocr_position = Variable(ocr_position.cuda(async=True))
                    ocr_ent = Variable(ocr_ent.cuda(async=True))
                    od = Variable(od.cuda(async=True))
                    od_mask = Variable(od_mask.cuda(async=True))
                    od_pos = Variable(od_pos.cuda(async=True))
                    od_position = Variable(od_position.cuda(async=True))
                    od_ent = Variable(od_ent.cuda(async=True))
                    ground_truth = Variable(ground_truth.cuda(async=True))
                    ground_truth_YN = Variable(ground_truth_YN.cuda(async=True))
                else:
                    q = Variable(q)
                    q_mask = Variable(q_mask)
                    q_pos = Variable(q_pos)
                    q_ent = Variable(q_ent)
                    ocr = Variable(ocr)
                    ocr_mask = Variable(ocr_mask)
                    ocr_pos = Variable(ocr_pos)
                    ocr_position = Variable(ocr_position)
                    ocr_ent = Variable(ocr_ent)
                    od = Variable(od)
                    od_mask = Variable(od_mask)
                    od_pos = Variable(od_pos)
                    od_position = Variable(od_position)
                    od_ent = Variable(od_ent)
                    ground_truth = Variable(ground_truth)
                    ground_truth_YN = Variable(ground_truth_YN)
                yield(img_fea, img_spa, od, od_mask, od_char, od_char_mask, od_pos, od_position, od_ent, od_bert, od_bert_mask, od_bert_offsets, q, q_mask, q_char, q_char_mask, q_pos, q_ent, q_bert, q_bert_mask, q_bert_offsets, ocr, ocr_mask, ocr_char, ocr_char_mask, ocr_pos, ocr_position, ocr_ent, ocr_bert, ocr_bert_mask, ocr_bert_offsets, od_offset, ocr_offset, od_last_index, ocr_last_index,od_max_num, ocr_max_num,  ground_truth, ground_truth_YN, ocr_ans_list, file_name, answers, question_id)


#===========================================================================
#=================== For standard evaluation in CoQA =======================
#===========================================================================

def ensemble_predict(pred_list, score_list, voteByCnt = False):
    predictions, best_scores = [], []
    pred_by_examples = list(zip(*pred_list))
    score_by_examples = list(zip(*score_list))
    for phrases, scores in zip(pred_by_examples, score_by_examples):
        d = defaultdict(float)
        firstappear = defaultdict(int)
        for phrase, phrase_score, index in zip(phrases, scores, range(len(scores))):
            d[phrase] += 1. if voteByCnt else phrase_score
            if not phrase in firstappear:
                firstappear[phrase] = -index
        predictions += [max(d.items(), key=lambda pair: (pair[1], firstappear[pair[0]]))[0]]
        best_scores += [max(d.items(), key=lambda pair: (pair[1], firstappear[pair[0]]))[1]]
    return (predictions, best_scores)

def _f1_score(pred, answers):
    def _score(g_tokens, a_tokens):
        common = Counter(g_tokens) & Counter(a_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1. * num_same / len(g_tokens)
        recall = 1. * num_same / len(a_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    if pred is None or answers is None:
        return 0

    if len(answers) == 0:
        return 1. if len(pred) == 0 else 0.
    
    g_tokens = _normalize_answer(pred).split()
    ans_tokens = [_normalize_answer(answer).split() for answer in answers]
    scores = [_score(g_tokens, a) for a in ans_tokens]
    if len(ans_tokens) == 1:
        score = scores[0]
    else:
        score = 0
        for i in range(len(ans_tokens)):
            scores_one_out = scores[:i] + scores[(i + 1):]
            score += max(scores_one_out)
        score /= len(ans_tokens)
    return score

def get_score(p, t, g, j):
    def normal_leven(str1, str2):
        str1 = str1.lower()
        str2 = str2.lower()
        len_str1 = len(str1) + 1
        len_str2 = len(str2) + 1
        #create matrix
        matrix = [0 for n in range(len_str1 * len_str2)]
        #init x axis
        for i in range(len_str1):
            matrix[i] = i
        #init y axis
        for j in range(0, len(matrix), len_str1):
            if j % len_str1 == 0:
                matrix[j] = j // len_str1
            
        for i in range(1, len_str1):
            for j in range(1, len_str2):
                if str1[i-1] == str2[j-1]:
                    cost = 0
                else:
                    cost = 1
                matrix[j*len_str1+i] = min(matrix[(j-1)*len_str1+i]+1,
                                            matrix[j*len_str1+(i-1)]+1,
                                            matrix[(j-1)*len_str1+(i-1)] + cost)
            
        return matrix[-1]
    def _get_s(p, t_g, g, j):
        t = []
        for _t in t_g:
            if _t == 'answering does not require reading text in the image':
                continue
            elif _t == 'unanswerable':
                t.append('')
            else:
                t.append(_t)
        if p == 'no_answer':
            flag = False
            for _t in t:
                if _t == '' or _t == '[]':
                    flag = True
            if flag:
                return 1, 1
            else:
                return 0, 0
        else:
            l = [max(len(_t), len(p)) for _t in t]
            s = [1 - normal_leven(p, _t) / _l for _t, _l in zip(t,l)]
            s_min = min(s)
            s_max = max(s)
            if s_max < 0.5:
                s_max = 0
            if s_min < 0.5:
                s_min = 0 
            return s_max, s_min
    # dis = _get_s(p, t, g, j)
    return _get_s(p, t, g, j)
def get_acc(g ,j):
    acc = 0
    if g == 0:
        if j['pred_idx'] == g[1]:
            acc = 1
        else:
            acc = 0
    else:
        if j['pred_idx'] == j['no_answer_idx']:
            acc = 1
        else:
            acc = 0
    return acc, 1 / j['answer_pool_len']

def score(pred, truth, gt, final_json):
    assert len(pred) == len(truth)
    no_ans_total = normal_total = total = 0
    no_ans_f1_min = no_ans_f1_max = normal_f1_max = normal_f1_min = f1_min = f1_max = 0
    all_f1 = []
    acc = 0
    acc_percentage = 0
    real_no_answer = 0
    for p, t, g, j in zip(pred, truth, gt, final_json):
        total += 1
        if g == 1:
            real_no_answer += 1
        this_f1_max, this_f1_min = get_score(p, t, g, j)
        # _acc, _acc_percentage = get_acc(g, j)
        # acc += _acc
        # acc_percentage += _acc_percentage
        f1_max += this_f1_max
        f1_min += this_f1_min
        all_f1.append([this_f1_max, this_f1_min])
        flag = False
        for _t in t:
            if _t == '' or _t == '[]' or _t == '{}':
                flag = True
        if flag:
            no_ans_total += 1
            no_ans_f1_max += this_f1_max
            no_ans_f1_min += this_f1_min
        else:
            normal_total += 1
            normal_f1_max += this_f1_max
            normal_f1_min += this_f1_min

    f1_max = f1_max / total
    f1_min = f1_min / total
    if no_ans_total == 0:
        no_ans_f1_max = no_ans_f1_min = 0.
    else:
        no_ans_f1_max = no_ans_f1_max / no_ans_total
        no_ans_f1_min = no_ans_f1_min / no_ans_total
    normal_f1_max = normal_f1_max / normal_total
    normal_f1_min = normal_f1_min / normal_total
    result = {
        'total': total,
        'f1': [f1_max, f1_min],
        'no_ans_total': no_ans_total,
        'no_ans_f1': [no_ans_f1_max, no_ans_f1_min],
        'normal_total': normal_total,
        'normal_f1': [normal_f1_max, normal_f1_min],
        'acc': acc / total,
        'base_acc': acc_percentage / total,
        'real_no_answer': [real_no_answer, real_no_answer / total]
    }
    return result, all_f1

def score_each_instance(pred, truth):
    assert len(pred) == len(truth)
    total = 0
    f1_scores = []
    for p, t in zip(pred, truth):
        total += 1
        f1_scores.append(_f1_score(p, t))
    f1_scores = [100. * x / total for x in f1_scores]
    return f1_scores

def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    

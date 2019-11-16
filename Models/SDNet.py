# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
from torch.nn.parameter import Parameter
from Models.Bert.Bert import Bert
from Models.Layers import MaxPooling, CNN, dropout, RNN_from_opt, set_dropout_prob, weighted_avg, set_seq_dropout, Attention, DeepAttention, LinearSelfAttn, GetFinalScores
from Utils.CoQAUtils import POS, ENT
from copy import deepcopy

'''
 SDNet
'''
class SDNet(nn.Module):
    def __init__(self, opt, word_embedding):
        super(SDNet, self).__init__()
        print('SDNet model\n')

        self.opt = opt
        #self.position_dim = opt['position_dim']
        self.use_cuda = (self.opt['cuda'] == True)
        set_dropout_prob(0.0 if not 'DROPOUT' in opt else float(opt['DROPOUT']))
        set_seq_dropout('VARIATIONAL_DROPOUT' in self.opt)

        x_input_size = 0
        ques_input_size = 0

        if 'GLOVE' in self.opt:
            self.vocab_size = int(opt['vocab_size'])
            vocab_dim = int(opt['vocab_dim'])
            self.vocab_embed = nn.Embedding(self.vocab_size, vocab_dim, padding_idx = 1)
            self.vocab_embed.weight.data = word_embedding
        else:
            self.vocab_size = 0
            self.opt['embedding_dim'] = 0
            self.opt['vocab_dim'] = 0
            vocab_dim = 0
        x_input_size += vocab_dim
        ques_input_size += vocab_dim

        if 'CHAR_CNN' in self.opt:
            print('CHAR_CNN')
            char_vocab_size = int(opt['char_vocab_size'])
            char_dim = int(opt['char_emb_size'])
            char_hidden_size = int(opt['char_hidden_size'])
            self.char_embed = nn.Embedding(char_vocab_size, char_dim, padding_idx = 1)
            self.char_cnn = CNN(char_dim, 3, char_hidden_size)
            self.maxpooling = MaxPooling()
            x_input_size += char_hidden_size
            ques_input_size += char_hidden_size

        if 'TUNE_PARTIAL' in self.opt:
            print('TUNE_PARTIAL')
            self.fixed_embedding = word_embedding[opt['tune_partial']:]
        elif 'GLOVE' in self.opt:    
            self.vocab_embed.weight.requires_grad = False

        cdim = 0
        self.use_contextual = False

        if 'BERT' in self.opt:
            print('Using BERT')
            self.Bert = Bert(self.opt)
            if 'LOCK_BERT' in self.opt:
                print('Lock BERT\'s weights')
                for p in self.Bert.parameters():
                    p.requires_grad = False
            if 'BERT_LARGE' in self.opt:
                print('BERT_LARGE')
                bert_dim = 1024
                bert_layers = 24
            else:
                bert_dim = 768
                bert_layers = 12

            print('BERT dim:', bert_dim, 'BERT_LAYERS:', bert_layers)    

            if 'BERT_LINEAR_COMBINE' in self.opt:
                print('BERT_LINEAR_COMBINE')
                self.alphaBERT = nn.Parameter(torch.Tensor(bert_layers), requires_grad=True)
                self.gammaBERT = nn.Parameter(torch.Tensor(1, 1), requires_grad=True)
                torch.nn.init.constant_(self.alphaBERT, 1.0)
                torch.nn.init.constant_(self.gammaBERT, 1.0)
                
            cdim = bert_dim
            x_input_size += bert_dim
            ques_input_size += bert_dim
        if 'PRE_ALIGN' in self.opt:
            self.pre_align = Attention(vocab_dim, opt['prealign_hidden'], correlation_func = 3, do_similarity = True)
            if 'PRE_ALIGN_befor_rnn' in self.opt:
                x_input_size += vocab_dim

        if 'pos_dim' in self.opt:
            pos_dim = opt['pos_dim']
            self.pos_embedding = nn.Embedding(len(POS), pos_dim)
            x_input_size += pos_dim
        if 'ent_dim' in self.opt:
            ent_dim = opt['ent_dim']
            self.ent_embedding = nn.Embedding(len(ENT), ent_dim)
            x_input_size += ent_dim

        # x_feat_len = 4
        # if 'ANSWER_SPAN_IN_CONTEXT_FEATURE' in self.opt:
        #     print('ANSWER_SPAN_IN_CONTEXT_FEATURE')
        #     x_feat_len += 1

        # x_input_size += pos_dim + ent_dim + x_feat_len
        

        print('Initially, the vector_sizes [doc, query] are', x_input_size, ques_input_size)
        addtional_feat = cdim if self.use_contextual else 0
        self.multi2one, multi2one_output_size = RNN_from_opt(x_input_size, opt['multi2one_hidden_size'],
            num_layers=1, concat_rnn=opt['concat_rnn'], add_feat=addtional_feat, bidirectional=self.opt['multi2one_bidir'])
        self.multi2one_output_size = multi2one_output_size
        if 'img_feature_replace_od' in self.opt:
            self.img_fea_num = self.opt['img_fea_num']
            self.img_fea_dim = self.opt['img_fea_dim']
            self.img_spa_dim = self.opt['img_spa_dim']
            self.img_fea2od = nn.Linear(self.opt['img_fea_dim'], multi2one_output_size)
            # self.pro_que_rnn, pro_que_rnn_output_size = RNN_from_opt(ques_input_size, multi2one_output_size//2)
            # assert pro_que_rnn_output_size == multi2one_output_size
            # ques_input_size = multi2one_output_size
        elif 'img_feature' in self.opt:
            self.img_fea_num = self.opt['img_fea_num']
            self.img_fea_dim = self.opt['img_fea_dim']
            self.img_spa_dim = self.opt['img_spa_dim']
            self.img_fea_linear = nn.Linear(self.opt['img_fea_dim'], opt['highlvl_hidden_size']*2)
            # self.img_fea_att = Attention(opt['highlvl_hidden_size']*2, opt['img_fea_att_hidden'], correlation_func=3)
            # self.pro_que_rnn, pro_que_rnn_output_size = RNN_from_opt(ques_input_size, multi2one_output_size//2)
            # assert pro_que_rnn_output_size == multi2one_output_size
            # ques_input_size = multi2one_output_size


        # RNN context encoder
        
        self.context_rnn, context_rnn_output_size = RNN_from_opt(multi2one_output_size, opt['hidden_size'],
            num_layers=opt['in_rnn_layers'], concat_rnn=opt['concat_rnn'], add_feat=addtional_feat)
        # RNN question encoder
        self.ques_rnn, ques_rnn_output_size = RNN_from_opt(ques_input_size, opt['hidden_size'],
            num_layers=opt['in_rnn_layers'], concat_rnn=opt['concat_rnn'], add_feat=addtional_feat)

        # Output sizes of rnn encoders
        print('After Input LSTM, the vector_sizes [doc, query] are [', context_rnn_output_size, ques_rnn_output_size, '] *', opt['in_rnn_layers'])

        # Deep inter-attention
        if 'GLOVE' not in self.opt:
            _word_hidden_size = 0
        else:
            _word_hidden_size = multi2one_output_size + addtional_feat

        self.deep_attn = DeepAttention(opt, abstr_list_cnt=opt['in_rnn_layers'], deep_att_hidden_size_per_abstr=opt['deep_att_hidden_size_per_abstr'], correlation_func=3, word_hidden_size=_word_hidden_size)

        self.deep_attn_input_size = self.deep_attn.rnn_input_size
        self.deep_attn_output_size = self.deep_attn.output_size
        print('Deep Attention: input: {}, hidden input: {}, output: {}'.format(self.deep_attn.att_size, self.deep_attn_input_size, self.deep_attn_output_size))

        # Question understanding and compression
        self.high_lvl_ques_rnn , high_lvl_ques_rnn_output_size = RNN_from_opt(ques_rnn_output_size * opt['in_rnn_layers'], opt['highlvl_hidden_size'], num_layers = opt['question_high_lvl_rnn_layers'], concat_rnn = True)

        self.after_deep_attn_size = self.deep_attn_output_size + self.deep_attn_input_size + addtional_feat + multi2one_output_size
        self.self_attn_input_size = self.after_deep_attn_size
                

        # Self attention on context
        if 'no_Context_Self_Attention' in self.opt:
            print('no self attention on context')
            self_attn_output_size = 0
        else:
            self.highlvl_self_att = Attention(self.self_attn_input_size, opt['deep_att_hidden_size_per_abstr'], correlation_func=3)
            self_attn_output_size = self.deep_attn_output_size
            print('Self deep-attention input is {}-dim'.format(self.self_attn_input_size))

        self.high_lvl_context_rnn, high_lvl_context_rnn_output_size = RNN_from_opt(self.deep_attn_output_size + self_attn_output_size, 
            opt['highlvl_hidden_size'], num_layers = 1, concat_rnn = False)
        context_final_size = high_lvl_context_rnn_output_size

        print('Do Question self attention')
        self.ques_self_attn = Attention(high_lvl_ques_rnn_output_size, opt['query_self_attn_hidden_size'], correlation_func=3)
        
        ques_final_size = high_lvl_ques_rnn_output_size
        print('Before answer span finding, hidden size are', context_final_size, ques_final_size)


        if 'position_dim' in self.opt:
            if self.opt['position_mod'] == 'qk+' or self.opt['position_mod'] == 'qk':
                self.od_ocr_attn = Attention(context_final_size, opt['hidden_size'], correlation_func = 3, do_similarity = True)
                self.position_attn = Attention(self.opt['position_dim'], opt['hidden_size'], correlation_func = 3, do_similarity = True)
                position_att_output_size = context_final_size
            elif self.opt['position_mod'] == 'cat':
                self.od_ocr_attn = Attention(context_final_size+self.opt['position_dim'], opt['hidden_size'], correlation_func = 3, do_similarity = True)
                position_att_output_size = context_final_size + self.opt['position_dim']
            elif self.opt['position_mod'] == 'noP':
                self.od_ocr_attn = Attention(context_final_size, opt['hidden_size'], correlation_func = 3, do_similarity = True)
                position_att_output_size = context_final_size
        # Question merging
        self.ques_merger = LinearSelfAttn(ques_final_size)
        if self.opt['pos_att_merge_mod'] == 'cat':
            self.get_answer = GetFinalScores(context_final_size + position_att_output_size, ques_final_size)
        elif self.opt['pos_att_merge_mod'] == 'atted':
            self.get_answer = GetFinalScores(position_att_output_size, ques_final_size)
        elif self.opt['pos_att_merge_mod'] == 'original':
            self.get_answer = GetFinalScores(context_final_size, ques_final_size)


    '''
    x: 1 x x_len (word_ids)
    x_single_mask: 1 x x_len
    x_char: 1 x x_len x char_len (char_ids)
    x_char_mask: 1 x x_len x char_len
    x_features: batch_size x x_len x feature_len (5, if answer_span_in_context_feature; 4 otherwise)
    x_pos: 1 x x_len (POS id)
    x_ent: 1 x x_len (entity id)
    x_bert: 1 x x_bert_token_len
    x_bert_mask: 1 x x_bert_token_len
    x_bert_offsets: 1 x x_len x 2
    q: batch x q_len  (word_ids)
    q_mask: batch x q_len
    q_char: batch x q_len x char_len (char ids)
    q_char_mask: batch x q_len x char_len
    q_bert: 1 x q_bert_token_len
    q_bert_mask: 1 x q_bert_token_len
    q_bert_offsets: 1 x q_len x 2
    context_len: number of words in context (only one per batch)
    return: 
      score_s: batch x context_len
      score_e: batch x context_len
      score_no: batch x 1
      score_yes: batch x 1
      score_noanswer: batch x 1
    '''
    def forward(self, img_fea, img_spa, x, x_mask, x_char, x_char_mask, x_pos, x_position, x_ent, x_bert, x_bert_mask, x_bert_offsets, q, q_mask, q_char, q_char_mask, q_pos, q_ent, q_bert, q_bert_mask, q_bert_offsets, o, o_mask, o_char, o_char_mask, o_pos, o_position, o_ent, o_bert, o_bert_mask, o_bert_offsets, od_offset, ocr_offset, od_last_index, ocr_last_index, od_max_num, ocr_max_num, return_score=False):
        att_score = {}
        if return_score:
            att_score['ocr_wid'] = deepcopy(x)
            att_score['od_wid'] = deepcopy(o)
            att_score['que_wid'] = deepcopy(q)
        batch_size = len(ocr_offset)
        x_input_list = []
        ques_input_list = []
        o_input_list = []
        if 'GLOVE' in self.opt:
            x_word_embed = self.vocab_embed(x) # batch x x_len x vocab_dim
            ques_word_embed = self.vocab_embed(q) # batch x q_len x vocab_dim
            o_word_embed = self.vocab_embed(o)

            x_input_list.append(dropout(x_word_embed, p=self.opt['dropout_emb'], training=self.drop_emb)) # batch x x_len x vocab_dim
            ques_input_list.append(dropout(ques_word_embed, p=self.opt['dropout_emb'], training=self.drop_emb)) # batch x q_len x vocab_dim
            o_input_list.append(dropout(o_word_embed, p=self.opt['dropout_emb'], training=self.drop_emb))

        # contextualized embedding
        x_cemb = ques_cemb = o_cemb = None        
        if 'BERT' in self.opt:
            x_cemb = ques_cemb = o_cemb = None
            
            if 'BERT_LINEAR_COMBINE' in self.opt:
                x_bert_output = self.Bert(x_bert, x_bert_mask, x_bert_offsets, x_mask)
                # print(x_single_mask.size(), x_bert.size())
                # print(x_single_mask.data)
                x_cemb_mid = self.linear_sum(x_bert_output, self.alphaBERT, self.gammaBERT)
                ques_bert_output = self.Bert(q_bert, q_bert_mask, q_bert_offsets, q_mask)
                # print(q_bert.size(), q_mask.size())
                # print(q_mask.data)
                ques_cemb_mid = self.linear_sum(ques_bert_output, self.alphaBERT, self.gammaBERT)
                o_bert_output = self.Bert(o_bert, o_bert_mask, o_bert_offsets, o_mask)
                o_cemb_mid = self.linear_sum(o_bert_output, self.alphaBERT, self.gammaBERT)
                # print(ques_cemb_mid.size())
                # print(x_cemb_mid.size())
                # x_cemb_mid = x_cemb_mid.expand(batch_size, -1, -1) &&&&&
                # print(x_cemb_mid.size())
            else:    
                x_cemb_mid = self.Bert(x_bert, x_bert_mask, x_bert_offsets, x_mask)
                # x_cemb_mid = x_cemb_mid.expand(batch_size, -1, -1) &&&&&
                ques_cemb_mid = self.Bert(q_bert, q_bert_mask, q_bert_offsets, q_mask)
                o_cemb_mid = self.Bert(o_bert, o_bert_mask, o_bert_offsets, o_mask)
            x_input_list.append(x_cemb_mid)
            ques_input_list.append(ques_cemb_mid)
            o_input_list.append(o_cemb_mid)

        if 'CHAR_CNN' in self.opt:
            x_char_final = self.character_cnn(x_char, x_char_mask)
            # x_char_final = x_char_final.expand(batch_size, -1, -1)
            ques_char_final = self.character_cnn(q_char, q_char_mask)
            o_char_final = self.character_cnn(o_char, o_char_mask)
            x_input_list.append(x_char_final)
            ques_input_list.append(ques_char_final)
            o_input_list.append(o_char_final)
        
        if 'PRE_ALIGN_befor_rnn' in self.opt:
            ocr_token_num_max = od_token_num_max = -1
            for i in range(batch_size):
                _od_num = 0
                for j in range(od_offset[i][0], od_offset[i][1]):
                    _od_num += od_last_index[j] + 1
                od_token_num_max = max(od_token_num_max, _od_num)
                _ocr_num = 0
                for j in range(ocr_offset[i][0], ocr_offset[i][1]):
                    _ocr_num += ocr_last_index[j] + 1
                ocr_token_num_max = max(ocr_token_num_max, _ocr_num)
            x_prealign_word_embed = torch.FloatTensor(batch_size,od_token_num_max,300).fill_(0).cuda()
            o_prealign_word_embed = torch.FloatTensor(batch_size, ocr_token_num_max, 300).fill_(0).cuda()
            for i in range(batch_size):
                od_cnt = 0
                for j in range(od_offset[i][0], od_offset[i][1]):
                    _j = od_last_index[j] + 1
                    x_prealign_word_embed[i][od_cnt:od_cnt+_j] = x_word_embed[j][:_j]
                    od_cnt += _j
                ocr_cnt = 0
                for j in range(ocr_offset[i][0], ocr_offset[i][1]):
                    _j = ocr_last_index[j] + 1
                    o_prealign_word_embed[i][ocr_cnt:ocr_cnt+_j] = o_word_embed[j][:_j]
                    ocr_cnt += _j
            x_prealign_glove = self.pre_align(x_prealign_word_embed, ques_word_embed, q_mask)
            o_prealign_glove = self.pre_align(o_prealign_word_embed, ques_word_embed, q_mask)

            x_prealign = torch.FloatTensor(x_word_embed.size(0), x_word_embed.size(1), x_prealign_glove.size(2)).fill_(0).cuda()
            o_prealign = torch.FloatTensor(o_word_embed.size(0), o_word_embed.size(1), o_prealign_glove.size(2)).fill_(0).cuda()
            for i in range(batch_size):
                od_cnt = 0
                for j in range(od_offset[i][0], od_offset[i][1]):
                    _j = od_last_index[j] + 1
                    x_prealign[j][:_j] = x_prealign_glove[i][od_cnt:od_cnt+_j]
                    od_cnt += _j
                ocr_cnt = 0
                for j in range(ocr_offset[i][0], ocr_offset[i][1]):
                    _j = ocr_last_index[j] + 1
                    o_prealign[j][:_j] = o_prealign_glove[i][ocr_cnt:ocr_cnt+_j]
                    ocr_cnt += _j

            x_input_list.append(x_prealign)
            o_input_list.append(o_prealign)
        if 'pos_dim' in self.opt:
            x_pos_emb = self.pos_embedding(x_pos)
            x_input_list.append(x_pos_emb)
            o_pos_emb = self.pos_embedding(o_pos)
            o_input_list.append(o_pos_emb)
        if 'ent_dim' in self.opt:
            x_ent_emb = self.ent_embedding(x_ent) # batch x x_len x ent_dim
            x_input_list.append(x_ent_emb)
            o_ent_emb = self.ent_embedding(o_ent) # batch x x_len x ent_dim
            o_input_list.append(o_ent_emb)

        _x_input = torch.cat(x_input_list, 2) # batch x x_len x (vocab_dim + cdim + vocab_dim + pos_dim + ent_dim + feature_dim)
        ques_input = torch.cat(ques_input_list, 2) # batch x q_len x (vocab_dim + cdim)
        _o_input = torch.cat(o_input_list, 2)
        multi2one_x_input = self.multi2one(_x_input, x_mask)
        multi2one_o_input = self.multi2one(_o_input, o_mask)
        if 'img_feature_replace_od' in self.opt:
            x_input = self.img_fea2od(img_fea.view(-1, self.img_fea_dim)).view(batch_size, self.img_fea_num, -1)
            x_mask = torch.ByteTensor(batch_size, self.img_fea_num).fill_(1).cuda()
        elif 'img_feature' in self.opt:
            img_fea = self.img_fea_linear(img_fea.view(-1, self.img_fea_dim)).view(batch_size, self.img_fea_num, -1)
            x_input = torch.FloatTensor(batch_size, od_max_num, self.multi2one_output_size).fill_(0).cuda()
            x_mask = torch.ByteTensor(batch_size, od_max_num).fill_(0).cuda()
            img_fea_mask = torch.ByteTensor(batch_size, self.img_fea_num).fill_(1).cuda()
        else:
            x_input = torch.FloatTensor(batch_size, od_max_num, self.multi2one_output_size).fill_(0).cuda()
            x_mask = torch.ByteTensor(batch_size, od_max_num).fill_(0).cuda()
        # x_input = torch.FloatTensor(batch_size, od_max_num, self.multi2one_output_size).fill_(0).cuda()
        o_input = torch.FloatTensor(batch_size, ocr_max_num, self.multi2one_output_size).fill_(0).cuda()
        # x_mask = torch.ByteTensor(batch_size, od_max_num).fill_(0).cuda()
        o_mask = torch.ByteTensor(batch_size, ocr_max_num).fill_(0).cuda()
        o_mask_pre = torch.ByteTensor(batch_size, ocr_max_num).fill_(1).cuda()
        for i in range(batch_size):
            if 'img_feature_replace_od' not in self.opt:
                od_cnt = 0
                for j in range(od_offset[i][0], od_offset[i][1]):
                    _j = od_last_index[j]
                    x_input[i][od_cnt] = multi2one_x_input[j][_j]
                    od_cnt += 1
                x_mask[i][0:od_cnt] = 1
            ocr_cnt = 0
            for j in range(ocr_offset[i][0], ocr_offset[i][1]):
                _j = ocr_last_index[j]
                o_input[i][ocr_cnt] = multi2one_o_input[j][_j]
                ocr_cnt += 1
            o_mask[i][0:ocr_cnt] = 1
            o_mask_pre[i][0:ocr_cnt-1] = 0
        # x_input.cuda()
        # o_input.cuda()
        # x_mask.cuda()
        # o_mask.cuda()
        # Multi-layer RNN
        if 'PRE_ALIGN_after_rnn' in self.opt:
            if return_score:
                x_prealign, x_word_leve_attention_score = self.pre_align(x_input, ques_word_embed, q_mask, return_score=return_score)
                o_prealign, o_word_leve_attention_score = self.pre_align(o_input, ques_word_embed, q_mask, return_score=return_score)
                att_score['WAa_ocr'] = x_word_leve_attention_score
                att_score['WAb_od'] = o_word_leve_attention_score
            else:
                x_prealign = self.pre_align(x_input, ques_word_embed, q_mask)
                o_prealign = self.pre_align(o_input, ques_word_embed, q_mask)

        _, x_rnn_layers = self.context_rnn(x_input, x_mask, return_list=True, x_additional=x_cemb) # layer x batch x x_len x context_rnn_output_size
        _, ques_rnn_layers = self.ques_rnn(ques_input, q_mask, return_list=True, x_additional=ques_cemb) # layer x batch x q_len x ques_rnn_output_size
        _, o_rnn_layers = self.context_rnn(o_input, o_mask, return_list=True, x_additional=o_cemb)

        # rnn with question only 
        ques_highlvl = self.high_lvl_ques_rnn(torch.cat(ques_rnn_layers, 2), q_mask) # batch x q_len x high_lvl_ques_rnn_output_size
        ques_rnn_layers.append(ques_highlvl) # (layer + 1) layers

        # deep multilevel inter-attention
        if x_cemb is None:
            if 'GLOVE' not in self.opt:
                x_long = []
                ques_long = []
                o_long = []
            elif 'PRE_ALIGN_after_rnn' in self.opt:
                x_long = [x_prealign]
                ques_long = [ques_word_embed]
                o_long = [o_prealign]
            else:
                x_long = [x_input]
                ques_long = [ques_word_embed]
                o_long = [o_input]
        else:
            x_long = [torch.cat([x_input, x_cemb], 2)]          # batch x x_len x (vocab_dim + cdim)
            ques_long = [torch.cat([ques_word_embed, ques_cemb], 2)] # batch x q_len x (vocab_dim + cdim)
            o_long = [torch.cat([o_input, o_cemb], 2)]
        if return_score:
            x_rnn_after_inter_attn, x_inter_attn, x_deep_att_score = self.deep_attn(x_long, x_rnn_layers, ques_long, ques_rnn_layers, x_mask, q_mask, return_bef_rnn=True, return_score=True)
            o_rnn_after_inter_attn, o_inter_attn, o_deep_att_score = self.deep_attn(o_long, o_rnn_layers, ques_long, ques_rnn_layers, o_mask, q_mask, return_bef_rnn=True, return_score=True)
            att_score['deep_att_ocr'] = x_deep_att_score
            att_score['deep_att_od'] = o_deep_att_score
        else:
            x_rnn_after_inter_attn, x_inter_attn = self.deep_attn(x_long, x_rnn_layers, ques_long, ques_rnn_layers, x_mask, q_mask, return_bef_rnn=True)
            o_rnn_after_inter_attn, o_inter_attn = self.deep_attn(o_long, o_rnn_layers, ques_long, ques_rnn_layers, o_mask, q_mask, return_bef_rnn=True)
        # x_rnn_after_inter_attn: batch x x_len x deep_attn_output_size
        # x_inter_attn: batch x x_len x deep_attn_input_size

        # deep self attention
        if x_cemb is None:
            x_self_attn_input = torch.cat([x_rnn_after_inter_attn, x_inter_attn, x_input], 2)
            o_self_attn_input = torch.cat([o_rnn_after_inter_attn, o_inter_attn, o_input], 2)
        else:
            x_self_attn_input = torch.cat([x_rnn_after_inter_attn, x_inter_attn, x_cemb, x_input], 2)
            o_self_attn_input = torch.cat([o_rnn_after_inter_attn, o_inter_attn, o_cemb, o_input], 2)
            # batch x x_len x (deep_attn_output_size + deep_attn_input_size + cdim + vocab_dim)
        
        if 'no_Context_Self_Attention' in self.opt:
            x_highlvl_output = self.high_lvl_context_rnn(x_rnn_after_inter_attn, x_mask)
            o_highlvl_output = self.high_lvl_context_rnn(o_rnn_after_inter_attn, o_mask)
        else:
            x_self_attn_output = self.highlvl_self_att(x_self_attn_input, x_self_attn_input, x_mask, x3=x_rnn_after_inter_attn, drop_diagonal=False)
            o_self_attn_output = self.highlvl_self_att(o_self_attn_input, o_self_attn_input, o_mask, x3=o_rnn_after_inter_attn, drop_diagonal=False)
            x_highlvl_output = self.high_lvl_context_rnn(torch.cat([x_rnn_after_inter_attn, x_self_attn_output], 2), x_mask)
            o_highlvl_output = self.high_lvl_context_rnn(torch.cat([o_rnn_after_inter_attn, o_self_attn_output], 2), o_mask)


        if 'position_dim' in self.opt:
            if 'img_feature_replace_od' in self.opt:
                    x_position = img_spa
            if self.opt['position_mod'] == 'qk+':
                if return_score:
                    x_od_ocr, x_od_ocr_att_score = self.od_ocr_attn(o_highlvl_output, x_highlvl_output, x_mask, return_score=return_score)
                    pos_att, pos_att_score = self.position_attn(o_position, x_position, x_mask, x3 = x_highlvl_output, return_score=return_score)
                    x_od_ocr += pos_att
                    att_score['od_ocr_att'] = x_od_ocr_att_score
                    att_score['pos_qk+'] = pos_att_score
                else:
                    x_od_ocr = self.od_ocr_attn(o_highlvl_output, x_highlvl_output, x_mask)
                    pos_att = self.position_attn(o_position, x_position, x_mask, x3 = x_highlvl_output)
                    x_od_ocr += pos_att
            elif self.opt['position_mod'] == 'qk':
                if return_score:
                    x_od_ocr, x_od_ocr_att_score = self.od_ocr_attn(o_highlvl_output, x_highlvl_output, x_mask, return_score=return_score)
                    pos_att, pos_att_score = self.position_attn(o_position, x_position, x_mask, x3 = x_highlvl_output, return_score=return_score)
                    # x_od_ocr += pos_att
                    att_score['od_ocr_att'] = x_od_ocr_att_score
                    att_score['pos_qk+'] = pos_att_score
                else:
                    x_od_ocr = self.od_ocr_attn(o_highlvl_output, x_highlvl_output, x_mask)
                    pos_att = self.position_attn(o_position, x_position, x_mask, x3 = x_highlvl_output)
                    # x_od_ocr += pos_att
            elif self.opt['position_mod'] == 'cat':
                if return_score:
                    x_od_ocr, x_od_ocr_att_score = self.od_ocr_attn(torch.cat([o_highlvl_output, o_position],dim=2), torch.cat([x_highlvl_output,x_position],dim=2), x_mask, return_score=return_score)
                    att_score['od_ocr_att'] = x_od_ocr_att_score
                else:
                    x_od_ocr = self.od_ocr_attn(torch.cat([o_highlvl_output, o_position],dim=2), torch.cat([x_highlvl_output,x_position],dim=2), x_mask)
            elif self.opt['position_mod'] == 'noP':
                if return_score:
                    x_od_ocr, x_od_ocr_att_score = self.od_ocr_attn(o_highlvl_output, x_highlvl_output, x_mask, return_score=return_score)
                    att_score['od_ocr_att'] = x_od_ocr_att_score
                else:
                    x_od_ocr = self.od_ocr_attn(o_highlvl_output, x_highlvl_output, x_mask)
        if self.opt['pos_att_merge_mod'] == 'cat':
            o_final = torch.cat([o_highlvl_output, x_od_ocr], 2)
        elif self.opt['pos_att_merge_mod'] == 'atted':
            o_final = x_od_ocr
        elif self.opt['pos_att_merge_mod'] == 'original':
            o_final = o_highlvl_output
        # question self attention  
        ques_final = self.ques_self_attn(ques_highlvl, ques_highlvl, q_mask, x3=None, drop_diagonal=False) # batch x q_len x high_lvl_ques_rnn_output_size

        # merge questions  
        q_merge_weights = self.ques_merger(ques_final, q_mask) 
        ques_merged = weighted_avg(ques_final, q_merge_weights) # batch x ques_final_size

        # predict scores
        score_s, score_noanswer = self.get_answer(o_final, ques_merged, o_mask)
        return score_s, score_noanswer, o_mask_pre, att_score
    '''
     input: 
      x_char: batch x word_num x char_num
      x_char_mask: batch x word_num x char_num
     output: 
       x_char_cnn_final:  batch x word_num x char_cnn_hidden_size
    '''
    def character_cnn(self, x_char, x_char_mask):
        x_char_embed = self.char_embed(x_char) # batch x word_num x char_num x char_dim
        batch_size = x_char_embed.shape[0]
        word_num = x_char_embed.shape[1]
        char_num = x_char_embed.shape[2]
        char_dim = x_char_embed.shape[3]
        x_char_cnn = self.char_cnn(x_char_embed.contiguous().view(-1, char_num, char_dim), x_char_mask) # (batch x word_num) x char_num x char_cnn_hidden_size
        x_char_cnn_final = self.maxpooling(x_char_cnn, x_char_mask.contiguous().view(-1, char_num)).contiguous().view(batch_size, word_num, -1) # batch x word_num x char_cnn_hidden_size
        return x_char_cnn_final

    def linear_sum(self, output, alpha, gamma):
        alpha_softmax = F.softmax(alpha, dim=0)
        for i in range(len(output)):
            t = output[i] * alpha_softmax[i] * gamma
            if i == 0:
                res = t
            else:
                res += t

        res = dropout(res, p=self.opt['dropout_emb'], training=self.drop_emb)
        return res

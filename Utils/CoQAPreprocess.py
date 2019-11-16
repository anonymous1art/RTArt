# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
    This file takes a CoQA data file as input and generates the input files for training the QA model.
"""
import json
import msgpack
import multiprocessing
import re
import string
import torch
from tqdm import tqdm
from collections import Counter
from Utils.GeneralUtils import nlp, load_glove_vocab, pre_proc
from Utils.CoQAUtils import token2id, token2id_sent, char2id_sent, build_embedding, feature_gen, POS, ENT
import os
import logging
from copy import deepcopy

log = logging.getLogger(__name__)
def normal_leven(str1, str2):
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

class CoQAPreprocess():
    def __init__(self, opt):
        print('CoQA Preprocessing')
        self.opt = opt
        self.spacyDir = opt['FEATURE_FOLDER']
        self.train_file = os.path.join(opt['datadir'], opt['TRAIN_FILE'])
        # self.dev_file = os.path.join(opt['datadir'], opt['CoQA_DEV_FILE'])
        self.glove_file = os.path.join(opt['datadir'], opt['INIT_WORD_EMBEDDING_FILE'])
        self.glove_dim = 300
        self.official = 'OFFICIAL' in opt
        self.data_prefix = 'vqa-'
        self.BuildTestVocabulary = 'BuildTestVocabulary' in opt
        self.n_gram = opt['n_gram']

        if self.official:
            #self.glove_vocab = load_glove_vocab(self.glove_file, self.glove_dim, to_lower = False)
            print('Official prediction initializes...')
            print('Loading training vocab and vocab char...')
            self.train_vocab, self.train_char_vocab, self.train_embedding = self.load_data()
            self.test_file = self.opt['OFFICIAL_TEST_FILE']
            # return

        # dataset_labels = ['train', 'dev']
        if self.official:
            dataset_labels = [self.opt['Task']]
        else:
            dataset_labels = ['train']
        allExist = True
        for dataset_label in dataset_labels:
            if not os.path.exists(os.path.join(self.spacyDir, self.data_prefix + dataset_label + '-preprocessed.msgpack')):
                allExist = False

        if allExist:
            return
        # else:
        #     assert False

        print('Previously result not found, creating preprocessed files now...')
        self.glove_vocab = load_glove_vocab(self.glove_file, self.glove_dim, to_lower = False)
        if not os.path.isdir(self.spacyDir):
            os.makedirs(self.spacyDir)
            print('Directory created: ' + self.spacyDir)
       
        for dataset_label in dataset_labels:
            self.preprocess(dataset_label)

    # dataset_label can be 'train' or 'dev' or 'test'
    def preprocess(self, dataset_label):
        file_name = self.train_file if dataset_label == 'train' else (self.dev_file if dataset_label == 'dev' else self.test_file)
        output_file_name = os.path.join(self.spacyDir, self.data_prefix + dataset_label + '-preprocessed.msgpack')
        log.info('Preprocessing : {}\n File : {}'.format(dataset_label, file_name))
        print('Loading json...')
        # with open(file_name, 'r') as f:
        #     dataset = json.load(f)
        with open(file_name, 'rb') as f:
            dataset = msgpack.load(f, encoding='utf8')
        if self.BuildTestVocabulary:
            test_id = {}
            test_file = self.opt['OFFICIAL_TEST_FILE']
            with open(test_file, 'r') as f:
                test_dataset = json.load(f)
            for item in test_dataset['data']:
                assert item['question_id'] not in test_id
                test_id[item['question_id']] = 1
            dataset['data'].extend(test_dataset['data'])
            output_file_name_test = os.path.join(self.spacyDir, self.data_prefix + self.opt['Task'] + '-preprocessed.msgpack')
        else:
            test_id = {}

        print('Processing json...')

        data = []
        # dataset['data'] = dataset['data'][:100] #debug
        tot = len(dataset['data'])
        ocr = 0
        od = 0
        yolo = 0
        quetion_str = []
        ans_str = []
        ocr_str = []
        od_str = []
        yolo_str = []

        ocr_dict = {}
        n_gram = self.n_gram

        #span for distinguishing the ocr distractors and 2-grams
        # len_ocr = []
        # len_2_gram = []

        over_range = []
        dis_pos_pad = [0 for i in range(8)]
        zero_len_ans = 0
        ocr_name_list_gram = ['OCR_gram2', 'TextSpotter_gram2', 'ensemble_ocr_gram2', 'two_stage_OCR_gram2', 'OCR_corner_gram2', 'PMTD_MORAN_gram2']
        ocr_name_list = ['distractors', 'OCR', 'TextSpotter', 'ensemble_ocr', 'two_stage_OCR', 'OCR_corner', 'PMTD_MORAN']
        od_name_list = ['OD', 'YOLO', 'OD_bottom-up']
        if 'preprocess_ocr_name' in self.opt:
            ocr_name_list = self.opt['preprocess_ocr_name'].split('#')
            ocr_name_list_gram = [t+'_gram'+str(self.opt['n_gram']) for t in ocr_name_list if t != 'distractors']
        if 'preprocess_od_name' in self.opt:
            od_name_list = self.opt['preprocess_od_name'].split('#')

        for data_idx in tqdm(range(tot)):
            datum = dataset['data'][data_idx]
            # dis_ocr = []
            # for _dis in datum['distractors']:
            #     if len(_dis) == 0:
            #         zero_len_ans += 1
            #         _dis = '#'
            #     dis_ocr.append({'word':_dis, 'pos':dis_pos_pad})
            # #assert len(dis_ocr) == 100
            # datum['distractors'] = dis_ocr
            if 'answers' not in datum:
                datum['answers'] = []
            # if len(datum['OCR']) == 0:
            #     continue

            
            que_str = datum['question'].lower()
            _datum = {'question': datum['question'],
                      'filename': datum['file_path'],
                      'question_id': datum['question_id'],
                      }
            for _ocr_name in ocr_name_list_gram:
                _datum['answers_id_'+_ocr_name] = datum['answers_id_'+_ocr_name]
            for _ocr_name in ocr_name_list:
                _datum['answers_id_'+_ocr_name] = datum['answers_id_'+_ocr_name]

            
            quetion_str.append(que_str)

            ans_str.append(' '.join([item.lower() for item in datum['answers']]))
            _datum['orign_answers'] = datum['answers']


            # _datum['OCR'] = []
            # _datum['distractors'] = []
            assert 'image_width' in datum
            assert 'image_height' in datum
            width = datum['image_width']
            height = datum['image_height']
            # ocr_name_list = ['distractors', 'OCR', 'ensemble_ocr', 'TextSpotter']
            for _ocr_name in ocr_name_list:
                _datum[_ocr_name] = []
                if _ocr_name not in datum:
                    datum[_ocr_name] = []
                for i in range(len(datum[_ocr_name])):
                    original = datum[_ocr_name][i]['word']
                    word = datum[_ocr_name][i]['word'].lower()
                    if word not in ocr_dict:
                        ocr_dict[word] = len(ocr_str)
                        ocr_str.append(datum[_ocr_name][i]['word'].lower())
                    ocr_pos = datum[_ocr_name][i]['pos']

                    for j in range(4):
                        ocr_pos[2*j] = ocr_pos[2*j] / width
                        ocr_pos[2*j+1] = ocr_pos[2*j+1] / height
                    for j in ocr_pos:
                        if not (j <= 1 and 0 <= j):
                            over_range.append(j)
                    _datum[_ocr_name].append({'word':word, 'pos': ocr_pos, 'original': original})
            for _od_name in od_name_list:
                _datum[_od_name] = []
                for i in range(len(datum[_od_name])):
                    original = datum[_od_name][i]['object']
                    od_str.append(datum[_od_name][i]['object'].lower())
                    _od_pos = datum[_od_name][i]['pos']
                    od_pos = []
                    _width = int(_od_pos[2] / 2)
                    _height = int(_od_pos[3] / 2)
                    od_pos.extend([_od_pos[0] - _width, _od_pos[1] - _height])
                    od_pos.extend([_od_pos[0] + _width, _od_pos[1] - _height])
                    od_pos.extend([_od_pos[0] + _width, _od_pos[1] + _height])
                    od_pos.extend([_od_pos[0] - _width, _od_pos[1] + _height])
                    for i in range(4):
                        od_pos[2*i] = od_pos[2*i] / width
                        od_pos[2*i+1] = od_pos[2*i+1] / height
                    for i in od_pos:
                        if not (i <= 1 and 0 <= i):
                            over_range.append(i)
                    _datum[_od_name].append({'object':[], 'pos': od_pos, 'original': original})
            data.append(_datum)
        # print('\nod num: {}\t ocr num: {}'.format(od, ocr))
        # log.info()
        log.info('ZERO LEGNTH ANS: {}'.format(zero_len_ans))
        log.info('length of data: {}'.format(len(data)))

        thread = multiprocessing.cpu_count()
        log.info('Using {} threads to takenize'.format(thread))
        que_iter = (pre_proc(c) for c in quetion_str)
        ans_iter = (pre_proc(c) for c in ans_str)
        ocr_iter = (pre_proc(c) for c in ocr_str)
        od_iter = (pre_proc(c) for c in od_str)
        yolo_iter = (pre_proc(c) for c in yolo_str)
        que_docs = [doc for doc in nlp.pipe(que_iter, batch_size=64, n_threads=thread)]
        ans_docs = [doc for doc in nlp.pipe(ans_iter, batch_size=64, n_threads=thread)]
        ocr_docs = [doc for doc in nlp.pipe(ocr_iter, batch_size=64, n_threads=thread)]
        od_docs = [doc for doc in nlp.pipe(od_iter, batch_size=64, n_threads=thread)]
        yolo_docs = [doc for doc in nlp.pipe(yolo_iter, batch_size=64, n_threads=thread)]
        assert len(que_docs) == len(quetion_str)
        assert len(ans_docs) == len(ans_str)
        assert len(ocr_docs) == len(ocr_str)
        assert len(od_docs) == len(od_str)
        assert len(yolo_docs) == len(yolo_str)
        ocr_output = [self.process(item) for item in ocr_docs]


        que_idx = ans_idx = ocr_idx = od_idx = yolo_idx = 0
        for _datum in tqdm(data):
            _datum['annotated_question'] = self.process(que_docs[que_idx])
            _datum['raw_question_offsets'] = self.get_raw_context_offsets(_datum['annotated_question']['word'], quetion_str[que_idx])
            que_idx += 1
            _datum['answers'] = self.process(ans_docs[ans_idx])
            ans_idx += 1
            for _ocr_name in ocr_name_list:
                for i in range(len(_datum[_ocr_name])):
                    # output = self.process(ocr_docs[ocr_idx])
                    # ocr_idx += 1
                    tmp_ocr = ocr_dict[_datum[_ocr_name][i]['word']]
                    if len(ocr_output[tmp_ocr]['word']) != 1:
                        ocr += 1
                    _datum[_ocr_name][i]['word'] = ocr_output[tmp_ocr]
                    ocr_idx += 1
            for _od_name in od_name_list:
                for i in range(len(_datum[_od_name])):
                    output = self.process(od_docs[od_idx])
                    od_idx += 1
                    if len(output['word']) != 1:
                        od += 1
                    _datum[_od_name][i]['object'] = output
        assert len(que_docs) == que_idx
        assert len(ans_docs) == ans_idx
        # assert len(ocr_docs) == ocr_idx
        assert len(od_docs) == od_idx
        assert len(yolo_docs) == yolo_idx
        log.info('od: {} \t ocr: {} \t yolo: {}'.format(od, ocr, yolo))
            

        # build vocabulary
        if dataset_label == 'train':
            print('Build vocabulary from training data...')
            contexts = [_datum['annotated_question']['word'] for _datum in data]
            for ocr_name in ocr_name_list:
                contexts.extend(
                    [item['word']['word'] for item in _datum[ocr_name] for _datum in data]
                )
            for od_name in od_name_list:
                contexts.extend(
                    [item['object']['word'] for item in _datum[od_name] for _datum in data]
                )
            # ocr = [item['word']['word'] for item in _datum['OCR'] for _datum in data]
            # od = [item['object']['word'] for item in _datum['OD'] for _datum in data]
            # yolo = [item['object']['word'] for item in _datum['YOLO'] for _datum in data]
            ans = [_datum['answers']['word'] for _datum in data]
            self.train_vocab = self.build_vocab(contexts, ans)
            self.train_char_vocab = self.build_char_vocab(self.train_vocab)

        print('Getting word ids...')
        w2id = {w: i for i, w in enumerate(self.train_vocab)}
        c2id = {c: i for i, c in enumerate(self.train_char_vocab)}
        que_oov = ocr_oov = 0
        od_oov = [0 for t in od_name_list]
        ocr_oov = [0 for t in ocr_name_list]
        od_token_total = [0 for t in od_name_list]
        ocr_token_total = [0 for t in ocr_name_list]
        que_token_total  = 0
        ocr_m1 = ocr_m2 = 0
        for _i, _datum in enumerate(data):
            _datum['annotated_question']['wordid'], oov, l = token2id_sent(_datum['annotated_question']['word'], w2id, unk_id = 1, to_lower = False)
            que_oov += oov
            que_token_total += l
            _datum['annotated_question']['charid'] = char2id_sent(_datum['annotated_question']['word'], c2id, unk_id = 1, to_lower = False)
            for _ocr_name_idx, _ocr_name in enumerate(ocr_name_list):
                for ocr_i, ocr in enumerate(_datum[_ocr_name]):
                    ocr['word']['wordid'], oov, l = token2id_sent(ocr['word']['word'], w2id, unk_id = 1, to_lower = False)
                    ocr_oov[_ocr_name_idx] += oov
                    ocr_token_total[_ocr_name_idx] += l
                    ocr['word']['charid'] = char2id_sent(ocr['word']['word'] , c2id, unk_id = 1, to_lower = False)
            for _od_name_idx, _od_name in enumerate(od_name_list):
                for ocr_i, ocr in enumerate(_datum[_od_name]):
                    ocr['object']['wordid'], oov, l = token2id_sent(ocr['object']['word'], w2id, unk_id = 1, to_lower = False)
                    od_oov[_od_name_idx] += oov
                    od_token_total[_od_name_idx] += l
                    ocr['object']['charid'] = char2id_sent(ocr['object']['word'] , c2id, unk_id = 1, to_lower = False)

            for _gram_name in ocr_name_list_gram:# 2 gram is wrong,changed by jin
                _datum[_gram_name] = []
                _ocr_name = _gram_name[:-6]
                n = int(_gram_name[-1])
                for i in range(len(_datum[_ocr_name])):
                    if i+n > len(_datum[_ocr_name]):
                        break
                    tmp = ' '.join([t['original'] for t in _datum[_ocr_name][i:i+n]])
                    word = {}
                    new_pos = []
                    for j in range(i, i+n):
                        if len(new_pos) == 0:
                            new_pos = deepcopy(_datum[_ocr_name][j]['pos'])
                        else:
                            for pos_idx in range(len(new_pos)):
                                if pos_idx == 0 or pos_idx == 1 or pos_idx == 3 or pos_idx == 4:
                                    new_pos[pos_idx] = min(new_pos[pos_idx], _datum[_ocr_name][j]['pos'][pos_idx])
                                else:
                                    new_pos[pos_idx] = max(new_pos[pos_idx], _datum[_ocr_name][j]['pos'][pos_idx])
                        for k, v in _datum[_ocr_name][j]['word'].items():
                            if k not in word:
                                word[k] = deepcopy(v)
                            else:
                                word[k] += deepcopy(v)
                    _datum[_gram_name].append({'word':word, 'pos': new_pos, 'original':tmp})
                for item in _datum[_gram_name]:
                    for wordid_item in item['word']['wordid']:
                        if type(wordid_item) is list:
                            assert False
        lines = [
            '|name|total token|oov|oov percentage|',
            '|:-:|:-:|:-:|:-:|'
        ]
        lines.append(
            '|question oov|{}|{}|{}|'.format(que_oov, que_token_total, que_oov/que_token_total)
        )
        print('question oov: {} / {} = {}'.format(que_oov, que_token_total, que_oov/que_token_total))
        for _ocr_name_idx, _ocr_name in enumerate(ocr_name_list):
            print('{} oov: {} / {} = {}'.format(_ocr_name, ocr_oov[_ocr_name_idx], ocr_token_total[_ocr_name_idx], ocr_oov[_ocr_name_idx]/ocr_token_total[_ocr_name_idx]))
            lines.append(
                '|{}|{}|{}|{}|'.format(_ocr_name, ocr_oov[_ocr_name_idx], ocr_token_total[_ocr_name_idx], ocr_oov[_ocr_name_idx]/ocr_token_total[_ocr_name_idx])
            )
        for _od_name_idx, _od_name in enumerate(od_name_list):
            print('{} oov: {} / {} = {}'.format(_od_name, od_oov[_od_name_idx], od_token_total[_od_name_idx], od_oov[_od_name_idx]/od_token_total[_od_name_idx]))
            lines.append(
                '|{}|{}|{}|{}|'.format(_od_name, od_oov[_od_name_idx], od_token_total[_od_name_idx], od_oov[_od_name_idx]/od_token_total[_od_name_idx])
            )
        with open(os.path.join(self.spacyDir, 'oov.md'), 'w') as f:
            f.write('\n'.join(lines))
        

        if dataset_label == 'train':
            # get the condensed dictionary embedding
            print('Getting embedding matrix for ' + dataset_label)
            embedding = build_embedding(self.glove_file, self.train_vocab, self.glove_dim)
            meta = {'vocab': self.train_vocab, 'char_vocab': self.train_char_vocab, 'embedding': embedding.tolist()}
            meta_file_name = os.path.join(self.spacyDir, dataset_label + '_meta.msgpack')
            print('Saving meta information to', meta_file_name)
            with open(meta_file_name, 'wb') as f:
                msgpack.dump(meta, f, encoding='utf8')
        if self.BuildTestVocabulary:
            test_dataset['data'] = data[-len(test_dataset['data']):]
            for item in test_dataset['data']:
                assert item['question_id'] in test_id
                test_id.pop(item['question_id'])
            assert 0 == len(test_id.keys())
            dataset['data'] = data[:-len(test_dataset['data'])]
            # with open(output_file_name_test, 'w') as output_file:
            #     json.dump(test_dataset, output_file, sort_keys=True, indent=2)
            with open(output_file_name_test, 'wb') as output_file:
                msgpack.dump(test_dataset, output_file, encoding='utf8')
        else:
            dataset['data'] = data

        # if dataset_label == 'test':
        #     return dataset

        # with open(output_file_name, 'w') as output_file:
        #     json.dump(dataset, output_file, sort_keys=True, indent=2)
        with open(output_file_name, 'wb') as output_file:
            msgpack.dump(dataset, output_file, encoding='utf8')
        log.info('Preprocessing over')
        
    '''
     Return train_vocab embedding
    '''
    def load_data(self):
        print('Load train_meta.msgpack...')
        meta_file_name = os.path.join(self.spacyDir, 'train_meta.msgpack')
        with open(meta_file_name, 'rb') as f:
            meta = msgpack.load(f, encoding='utf8')
        embedding = torch.Tensor(meta['embedding'])
        self.opt['vocab_size'] = embedding.size(0)
        self.opt['vocab_dim'] = embedding.size(1)
        self.opt['char_vocab_size'] = len(meta['char_vocab'])
        return meta['vocab'], meta['char_vocab'], embedding

    def build_vocab(self, contexts, qas): # vocabulary will also be sorted accordingly
        counter_c = Counter(w for doc in contexts for w in doc)
        counter_qa = Counter(w for doc in qas for w in doc)
        counter = counter_c + counter_qa
        vocab = sorted([t for t in counter_qa if t in self.glove_vocab], key=counter_qa.get, reverse=True)
        vocab += sorted([t for t in counter_c.keys() - counter_qa.keys() if t in self.glove_vocab],
                        key=counter.get, reverse=True)
        total = sum(counter.values())
        matched = sum(counter[t] for t in vocab)
        print('vocab {1}/{0} OOV {2}/{3} ({4:.4f}%)'.format(
            len(counter), len(vocab), (total - matched), total, (total - matched) / total * 100))
        vocab.insert(0, "<PAD>")
        vocab.insert(1, "<UNK>")
        vocab.insert(2, "<Q>")
        vocab.insert(3, "<OCR>")
        vocab.insert(4, "<OD>")
        return vocab

    def build_char_vocab(self, words):
        counter = Counter(c for w in words for c in w)
        print('All characters: {0}'.format(len(counter)))
        char_vocab = [c for c, cnt in counter.items() if cnt > 3]
        print('Occurrence > 3 characters: {0}'.format(len(char_vocab)))

        char_vocab.insert(0, "<PAD>")
        char_vocab.insert(1, "<UNK>")
        char_vocab.insert(2, "<STA>")
        char_vocab.insert(3, "<END>")
        return char_vocab    

    def _str(self, s):
        """ Convert PTB tokens to normal tokens """
        if (s.lower() == '-lrb-'):
            s = '('
        elif (s.lower() == '-rrb-'):
            s = ')'
        elif (s.lower() == '-lsb-'):
            s = '['
        elif (s.lower() == '-rsb-'):
            s = ']'
        elif (s.lower() == '-lcb-'):
            s = '{'
        elif (s.lower() == '-rcb-'):
            s = '}'
        return s

    def process(self, parsed_text):
        output = {'word': [],
                  'lemma': [],
                  'pos': [],
                  'pos_id': [],
                  'ent': [],
                  'ent_id': [],
                  'offsets': [],
                  'sentences': []}

        for token in parsed_text:
            #[(token.text,token.idx) for token in parsed_sentence]
            output['word'].append(self._str(token.text))
            pos = token.tag_
            output['pos'].append(pos)
            output['pos_id'].append(token2id(pos, POS, 0))

            ent = 'O' if token.ent_iob_ == 'O' else (token.ent_iob_ + '-' + token.ent_type_)
            output['ent'].append(ent)
            output['ent_id'].append(token2id(ent, ENT, 0))

            output['lemma'].append(token.lemma_ if token.lemma_ != '-PRON-' else token.text.lower())
            output['offsets'].append((token.idx, token.idx + len(token.text)))

        word_idx = 0
        for sent in parsed_text.sents:
            output['sentences'].append((word_idx, word_idx + len(sent)))
            word_idx += len(sent)

        assert word_idx == len(output['word'])
        return output

    '''
     offsets based on raw_text
     this will solve the problem that, in raw_text, it's "a-b", in parsed test, it's "a - b"
    '''
    def get_raw_context_offsets(self, words, raw_text):
        raw_context_offsets = []
        p = 0
        for token in words:            
            while p < len(raw_text) and re.match('\s', raw_text[p]):
                p += 1
            if raw_text[p:p + len(token)] != token:
                print('something is wrong! token', token, 'raw_text:', raw_text)

            raw_context_offsets.append((p, p + len(token)))
            p += len(token)

        return raw_context_offsets

    def normalize_answer(self, s):
        """Lower text and remove punctuation, storys and extra whitespace."""

        def remove_articles(text):
            regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
            return re.sub(regex, ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))


    # find the word id start and stop
    def find_span_with_gt(self, context, offsets, ground_truth):
        best_f1 = 0.0
        best_span = (len(offsets) - 1, len(offsets) - 1)
        gt = self.normalize_answer(pre_proc(ground_truth)).split()

        ls = [i for i in range(len(offsets)) if context[offsets[i][0]:offsets[i][1]].lower() in gt]

        for i in range(len(ls)):
            for j in range(i, len(ls)):
                pred = self.normalize_answer(pre_proc(context[offsets[ls[i]][0]: offsets[ls[j]][1]])).split()
                common = Counter(pred) & Counter(gt)
                num_same = sum(common.values())
                if num_same > 0:
                    precision = 1.0 * num_same / len(pred)
                    recall = 1.0 * num_same / len(gt)
                    f1 = (2 * precision * recall) / (precision + recall)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_span = (ls[i], ls[j])
        return best_span


    # find the word id start and stop
    def find_span(self, offsets, start, end):
        start_index = -1
        end_index = -1
        for i, offset in enumerate(offsets):
            if (start_index < 0) or (start >= offset[0]):
                start_index = i
            if (end_index < 0) and (end <= offset[1]):
                end_index = i
        return (start_index, end_index)

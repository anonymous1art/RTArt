from datetime import datetime
import json
import numpy as np
import os
import msgpack
import random
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Utils.CoQAPreprocess import CoQAPreprocess
from Models.Layers import MaxPooling, set_dropout_prob
from Models.SDNet import SDNet
from Models.BaseTrainer import BaseTrainer
from Utils.CoQAUtils import BatchGen, AverageMeter, gen_upper_triangle, score
import pickle, h5py
from tqdm import tqdm
 
class SDNetTrainer(BaseTrainer):
    def __init__(self, opt):
        super(SDNetTrainer, self).__init__(opt)
        print('SDNet Model Trainer')
        self.opt = opt
        set_dropout_prob(0.0 if not 'DROPOUT' in opt else float(opt['DROPOUT']))
        self.seed = int(opt['SEED'])
        self.data_prefix = 'vqa-'
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.preproc = CoQAPreprocess(self.opt)
        if self.use_cuda:
            torch.cuda.manual_seed_all(self.seed)

    def official(self, model_path, test_data):
        print('-----------------------------------------------')
        print("Initializing model...")
        self.setup_model(self.preproc.train_embedding)
        self.load_model(model_path)

        print("Predicting in batches...")
        test_batches = BatchGen(self.opt, test_data['data'], self.use_cuda, self.preproc.train_vocab, self.preproc.train_char_vocab, evaluation=True)
        predictions = []
        confidence = []
        final_json = []
        cnt = 0
        for j, test_batch in enumerate(test_batches):
            cnt += 1
            if cnt % 50 == 0:
                print(cnt, '/', len(test_batches))  
            phrase, phrase_score, pred_json = self.predict(test_batch)
            predictions.extend(phrase)
            confidence.extend(phrase_score)
            final_json.extend(pred_json)

        return predictions, confidence, final_json

    def train(self): 
        self.isTrain = True
        self.getSaveFolder()
        self.saveConf()
        self.vocab, self.char_vocab, vocab_embedding = self.preproc.load_data()
        self.log('-----------------------------------------------')
        self.log("Initializing model...")
        self.setup_model(vocab_embedding)
        self.reture_att_score = 'att_score' in self.opt
        
        if 'RESUME' in self.opt:
            model_path = os.path.join(self.opt['datadir'], self.opt['MODEL_PATH'])
            self.load_model(model_path)            

        print('Loading train json...')
        with open(os.path.join(self.opt['FEATURE_FOLDER'], self.data_prefix + 'train-preprocessed.msgpack'), 'rb') as f:
            train_data = msgpack.load(f, encoding='utf8')
        print('Dataset has been loaded')

        # print('Loading dev json...')
        # with open(os.path.join(self.opt['FEATURE_FOLDER'], self.data_prefix + 'dev-preprocessed.json'), 'r') as f:
        #     dev_data = json.load(f)

        best_f1_max = best_f1_min = -1
        best_acc = -1
        numEpochs = self.opt['EPOCH']
        indices = list(range(len(train_data['data'])))
        random.shuffle(indices)
        train_data['data'] = [train_data['data'][i] for i in indices]
        # train_img_id2idx = pickle.load(open(os.path.join(self.opt['FEATURE_FOLDER'], 'train36_imgid2idx.pkl'), 'rb'))
        # val_img_id2idx = pickle.load(open(os.path.join(self.opt['FEATURE_FOLDER'], 'val36_imgid2idx.pkl'), 'rb'))
        if 'img_feature' in self.opt:
            print('Loading Image Feature')
            train_img_id2idx = pickle.load(open(os.path.join(self.opt['img_feature_dir'], 'train36_imgid2idx.pkl'), 'rb'))
            val_img_id2idx = pickle.load(open(os.path.join(self.opt['img_feature_dir'], 'val36_imgid2idx.pkl'), 'rb'))
            with h5py.File(os.path.join(self.opt['img_feature_dir'], 'train36.hdf5'), 'r') as hf:
                train_img_features = torch.tensor(hf.get('image_features'))
                train_img_spatials = torch.tensor(hf.get('spatial_features'))
            with h5py.File(os.path.join(self.opt['img_feature_dir'], 'val36.hdf5'), 'r') as hf:
                val_img_features = torch.tensor(hf.get('image_features'))
                val_img_spatials = torch.tensor(hf.get('spatial_features'))
            print('Image Feature Loaded')
        else:
            train_img_id2idx = val_img_id2idx = train_img_features = train_img_spatials = val_img_features = val_img_spatials = None
        for epoch in range(self.epoch_start, numEpochs):
            # self.log('Epoch {}'.format(epoch))
            self.network.train()
            startTime = datetime.now()
            self.opt['current_epoch'] = epoch
            # train_batches = BatchGen(self.opt, train_data['data'], self.use_cuda, self.vocab, self.char_vocab, mod='train')
            # dev_batches = BatchGen(self.opt, train_data['data'], self.use_cuda, self.vocab, self.char_vocab, mod='dev')
            train_batches = BatchGen(self.opt, train_data['data'], self.use_cuda, self.vocab, self.char_vocab, train_img_id2idx, train_img_features, train_img_spatials, val_img_id2idx, val_img_features, val_img_spatials, mod='train')
            dev_batches = BatchGen(self.opt, train_data['data'], self.use_cuda, self.vocab, self.char_vocab, train_img_id2idx,train_img_features, train_img_spatials, val_img_id2idx, val_img_features, val_img_spatials, mod='dev')
            train_batch_len = len(train_batches)



            # val_dic = {}
            # error_num = 0
            # # train_batch_len = 4
            # for datum_i, datum in enumerate(train_data['data']):
            #     q_id = datum['question_id']
            #     if q_id not in val_dic:
            #         val_dic[q_id] = [datum_i, len(datum['OD']), len(datum['OCR'])]
            #     else:
            #         val_dic[q_id].extend([datum_i, len(datum['OD']), len(datum['OCR'])])
            #         error_num += 1
            # val_dic['error_num'] = error_num
            # with open('val_dic_epoch_{}.json'.format(epoch), 'w') as f:
            #     json.dump(val_dic, f, indent=2)



            for i, batch in enumerate(train_batches):
                if (i == 0 and epoch == 0) or (epoch == 0 and i == 0 and ('RESUME' in self.opt)) or (i > 0 and i % 1500 == 0) or i == train_batch_len - 1:
                    # print('Saving folder is', self.saveFolder)
                    # print('Evaluating on dev set...')
                    predictions = []
                    confidence = []
                    dev_answer = []
                    final_json = []
                    GT = []
                    with torch.no_grad():
                        for j, dev_batch in enumerate(dev_batches):
                            phrase, phrase_score, pred_json = self.predict(dev_batch)
                            final_json.extend(pred_json)
                            predictions.extend(phrase)
                            confidence.extend(phrase_score)
                            dev_answer.extend(dev_batch[-2]) # answer_str
                            GT.extend(dev_batch[-5].detach().cpu().numpy().tolist())
                    if self.reture_att_score:
                        pred_json_file = os.path.join(self.saveFolder, 'epoch_{}_i_{}_prediction.json'.format(epoch, i))
                        with open(pred_json_file, 'w') as output_file:
                            json.dump(final_json, output_file, indent=2)
                    result, all_f1s = score(predictions, dev_answer, GT, final_json)
                    f1_max = result['f1'][0]
                    f1_min = result['f1'][1]
                    acc = result['acc']
                    if f1_max > best_f1_max:
                        model_file = os.path.join(self.saveFolder, 'f1_max_best_model.pt')
                        self.save_for_predict(model_file, epoch)
                        best_f1_max = f1_max
                        pred_json_file = os.path.join(self.saveFolder, 'f1_max_prediction.json')
                        with open(pred_json_file, 'w') as output_file:
                            json.dump(final_json, output_file, indent=2)
                        score_per_instance = []    
                        for instance, s in zip(final_json, all_f1s):
                            score_per_instance.append({
                                'filename': instance['file_name'],
                                'f1': s
                            })
                        score_per_instance_json_file = os.path.join(self.saveFolder, 'f1_max_score_per_instance.json')
                        with open(score_per_instance_json_file, 'w') as output_file:
                            json.dump(score_per_instance, output_file, indent=2)
                    if f1_min > best_f1_min:
                        model_file = os.path.join(self.saveFolder, 'f1_min_best_model.pt')
                        self.save_for_predict(model_file, epoch)
                        best_f1_min = f1_min
                        pred_json_file = os.path.join(self.saveFolder, 'f1_min_prediction.json')
                        with open(pred_json_file, 'w') as output_file:
                            json.dump(final_json, output_file, indent=2)
                        score_per_instance = []    
                        for instance, s in zip(final_json, all_f1s):
                            score_per_instance.append({
                                'filename': instance['file_name'],
                                'f1': s
                            })
                        score_per_instance_json_file = os.path.join(self.saveFolder, 'f1_min_score_per_instance.json')
                        with open(score_per_instance_json_file, 'w') as output_file:
                            json.dump(score_per_instance, output_file, indent=2)     

                    self.log("Epoch {0} - dev: F1_max: {1:.3f} (best F1_max: {2:.3f}  F1_min: {7:.3f} (best F1_min: {8:.3f}) -dev: ACC: {3:.3f} (best ACC: {4:.3f}, base ACC: {5:.3f}, no_ans ACC: {6:.3f})".format(epoch, f1_max, best_f1_max, acc, best_acc, result['base_acc'], result['real_no_answer'][1], f1_min, best_f1_min))
                    # assert False
                    # self.log("Results breakdown\n{0}".format(result))
                
                self.update(batch)
                if i % 30 == 0:
                    self.log('updates[{0:6}] train loss[{1:.5f}] remaining[{2}]'.format(
                        self.updates, self.train_loss.avg,
                        str((datetime.now() - startTime) / (i + 1) * (len(train_batches) - i - 1)).split('.')[0]))

            # print("PROGRESS: {0:.2f}%".format(100.0 * (epoch + 1) / numEpochs))
            # print('Config file is at ' + self.opt['confFile'])
    def predict_for_dev(self): 
        self.isTrain = False
        self.getSaveFolder()
        self.vocab, self.char_vocab, vocab_embedding = self.preproc.load_data()
        print('-----------------------------------------------')
        print("Initializing model...")
        self.setup_model(vocab_embedding)
        
        if 'RESUME' in self.opt:
            model_path = os.path.join(self.opt['datadir'], self.opt['MODEL_PATH'])
            self.load_model(model_path)            

        print('Loading train json...')
        with open(os.path.join(self.opt['FEATURE_FOLDER'], self.data_prefix + 'train-preprocessed.json'), 'r') as f:
            train_data = json.load(f)

        # print('Loading dev json...')
        # with open(os.path.join(self.opt['FEATURE_FOLDER'], self.data_prefix + 'dev-preprocessed.json'), 'r') as f:
        #     dev_data = json.load(f)



        # self.network.train()
        # indices = list(range(len(train_data['data'])))
        # random.shuffle(indices)
        # train_data['data'] = [train_data['data'][i] for i in indices]
        dev_batches = BatchGen(self.opt, train_data['data'], self.use_cuda, self.vocab, self.char_vocab, mod='dev')
        predictions = []
        confidence = []
        dev_answer = []
        final_json = []
        GT = []
        for j, dev_batch in enumerate(dev_batches):
            phrase, phrase_score, pred_json = self.predict(dev_batch)
            final_json.extend(pred_json)
            predictions.extend(phrase)
            confidence.extend(phrase_score)
            dev_answer.extend(dev_batch[-2]) # answer_str
            GT.extend(dev_batch[-5].data.cpu().numpy().tolist())
        result, all_f1s = score(predictions, dev_answer, GT, final_json)
        f1_max = result['f1'][0]
        f1_min = result['f1'][1]
        acc = result['acc']
        model_name = 'dev_' + self.opt['MODEL_PATH'].split('/')[-1][:-3]

        pred_json_file = os.path.join(self.saveFolder, model_name + '_prediction.json')
        with open(pred_json_file, 'w') as output_file:
            json.dump(final_json, output_file, indent=2)
        score_per_instance = []    
        for instance, s in zip(final_json, all_f1s):
            score_per_instance.append({
                'filename': instance['file_name'],
                'f1': s
            })
        score_per_instance_json_file = os.path.join(self.saveFolder, model_name + '_score.json')
        with open(score_per_instance_json_file, 'w') as output_file:
            json.dump(score_per_instance, output_file, indent=2)

        print("dev: F1_max: {0:.3f} F1_min: {1:.3f} ACC: {2:.3f} (base ACC: {3:.3f}, no_ans ACC: {4:.3f})".format(f1_max, f1_min, acc, result['base_acc'], result['real_no_answer'][1]))
        # self.log("Results breakdown\n{0}".format(result))
    def predict_for_test(self): 
        self.reture_att_score = 'att_score' in self.opt#jin
        self.isTrain = False
        self.getSaveFolder()
        self.vocab, self.char_vocab, vocab_embedding = self.preproc.load_data()
        print('-----------------------------------------------')
        print("Initializing model...")
        self.setup_model(vocab_embedding)
        
        print('Loading train json...')
        with open(os.path.join(self.opt['FEATURE_FOLDER'], self.data_prefix + self.opt['Task'] + '-preprocessed.msgpack'), 'rb') as f:
            train_data = msgpack.load(f, encoding='utf8')

        if 'RESUME' in self.opt:
            model_path = os.path.join(self.opt['datadir'], self.opt['MODEL_PATH'])
            self.load_model(model_path)            

        

        # print('Loading dev json...')
        # with open(os.path.join(self.opt['FEATURE_FOLDER'], self.data_prefix + 'dev-preprocessed.json'), 'r') as f:
        #     dev_data = json.load(f)



        # self.network.train()
        # indices = list(range(len(train_data['data'])))
        # random.shuffle(indices)
        # train_data['data'] = [train_data['data'][i] for i in indices]
        #dev_batches = BatchGen(self.opt, train_data['data'], self.use_cuda, self.vocab, self.char_vocab, mod='test')#jin
        if 'img_feature' in self.opt:
            train_img_id2idx = pickle.load(open(os.path.join(self.opt['img_feature_dir'], 'task3_test_imgid2idx.pkl'), 'rb'))
            with h5py.File(os.path.join(self.opt['img_feature_dir'], 'task3_test.hdf5'), 'r') as hf:
                train_img_features = torch.tensor(hf.get('image_features'))
                train_img_spatials = torch.tensor(hf.get('spatial_features'))
        else:
            train_img_id2idx = val_img_id2idx = train_img_features = train_img_spatials = val_img_features = val_img_spatials = None
        # train_img_id2idx = val_img_id2idx = train_img_features = train_img_spatials = val_img_features = val_img_spatials = None#jin
        dev_batches = BatchGen(self.opt, train_data['data'], self.use_cuda, self.vocab, self.char_vocab, train_img_id2idx,train_img_features, train_img_spatials, None, None, None, mod='test')#jin
        # predictions = []
        # confidence = []
        # dev_answer = []
        # GT = []
        final_json = []
        # print()
        for dev_batch in tqdm(dev_batches):
            phrase, phrase_score, pred_json = self.predict(dev_batch)
            final_json.extend(pred_json)
            # predictions.extend(phrase)
            # confidence.extend(phrase_score)
            # dev_answer.extend(dev_batch[-2]) # answer_str
            # GT.extend(dev_batch[-5].data.cpu().numpy().tolist())
        model_name = 'test_' + self.opt['MODEL_PATH'].split('/')[-1][:-3]

        pred_json_file = os.path.join(self.saveFolder, model_name + '_prediction_all.json')
        with open(pred_json_file, 'w') as output_file:
            json.dump(final_json, output_file, indent=2)
        submission = []
        no_ans_cnt = 0    
        for item in final_json:
            _s = {}
            if item['answer'] == 'no_answer':
                _s['answer'] = ''
                no_ans_cnt += 1
            else:
                _s['answer'] = item['answer']
            _s['question_id'] = item['question_id']
            if item['question_id'] >= 34601:
                continue
            submission.append(_s)
        print('Test Samples Number: ', len(submission))
        assert len(submission) == 4070
        submission_json_file = os.path.join(self.saveFolder, model_name + '_submission.json')
        with open(submission_json_file, 'w') as output_file:
            json.dump(submission, output_file, indent=2)
        print('no answer: {} / {} percentage: {}'.format(no_ans_cnt, len(train_data['data']), no_ans_cnt / len(train_data['data'])))

        print("submission file is complited")
        # self.log("Results breakdown\n{0}".format(result))

    def setup_model(self, vocab_embedding):
        self.train_loss = AverageMeter()
        self.network = SDNet(self.opt, vocab_embedding)
        # self.network = nn.DataParallel(SDNet(self.opt, vocab_embedding))
        if self.use_cuda:
            self.log('Putting model into GPU')
            self.network.cuda()

        parameters = [p for p in self.network.parameters() if p.requires_grad]
        self.optimizer = optim.Adamax(parameters)
        if 'ADAM2' in self.opt:
            print('ADAM2')
            self.optimizer = optim.Adam(parameters, lr = 0.0001)

        self.updates = 0
        self.epoch_start = 0
        if self.opt['loss'] == 'binaryCE':
            self.loss_func = self.instance_bce_with_logits
        elif self.opt['loss'] == 'CE':
            self.loss_func = F.cross_entropy
        else:
            print('loss parameter is error')
            assert False

    def update(self, batch):
        # Train mode
        self.network.train()
        self.network.drop_emb = True

        img_fea, img_spa, od, od_mask, od_char, od_char_mask, od_pos, od_position, od_ent, od_bert, od_bert_mask, od_bert_offsets, q, q_mask, q_char, q_char_mask, q_pos, q_ent, q_bert, q_bert_mask, q_bert_offsets, ocr, ocr_mask, ocr_char, ocr_char_mask, ocr_pos, ocr_position, ocr_ent, ocr_bert, ocr_bert_mask, ocr_bert_offsets, od_offset, ocr_offset, od_last_index, ocr_last_index,od_max_num, ocr_max_num,  ground_truth, ground_truth_YN, ocr_ans_list, file_name, answers, question_id = batch

        # Run forward
        # score_s, score_e: batch x context_word_num
        # score_yes, score_no, score_no_answer: batch x 1
        score_s, score_no_answer, mask, _ = self.network(img_fea, img_spa, od, od_mask, od_char, od_char_mask, od_pos, od_position, od_ent, od_bert, od_bert_mask, od_bert_offsets, q, q_mask, q_char, q_char_mask, q_pos, q_ent, q_bert, q_bert_mask, q_bert_offsets, ocr, ocr_mask, ocr_char, ocr_char_mask, ocr_pos, ocr_position, ocr_ent, ocr_bert, ocr_bert_mask, ocr_bert_offsets, od_offset, ocr_offset, od_last_index, ocr_last_index,od_max_num, ocr_max_num)
        score_s = F.softmax(score_s, dim=1)
        scores = torch.cat((score_s, score_no_answer), dim=1) # batch x (context_len * context_len + 3)
        targets = torch.cat([ground_truth, ground_truth_YN.unsqueeze(1)], dim=1)
        if self.opt['loss'] == 'CE':
            targets = torch.nonzero(targets)[:,1]

        # span_idx = score_s.shape[1]
        # for i in range(ground_truth_YN.size(0)):
        #     if ground_truth_YN[i] == -1:  # no answer
        #         targets[i][span_idx] = 1
        #     else:
        #         targets[i][ground_truth[i]] = 1
        # if self.use_cuda:
        #     targets = targets.cuda()
        # loss = self.instance_bce_with_logits(scores, targets.float())
        loss = self.loss_func(scores, targets)
        self.train_loss.update(loss.item(), 1)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.opt['grad_clipping'])
        self.optimizer.step()
        self.updates += 1
        if 'TUNE_PARTIAL' in self.opt:
            self.network.vocab_embed.weight.data[self.opt['tune_partial']:] = self.network.fixed_embedding
        # torch.cuda.empty_cache()

    def predict(self, batch, all_ans=False):
        self.network.eval()
        self.network.drop_emb = False

        # Run forward
        img_fea, img_spa, od, od_mask, od_char, od_char_mask, od_pos, od_position, od_ent, od_bert, od_bert_mask, od_bert_offsets, q, q_mask, q_char, q_char_mask, q_pos, q_ent, q_bert, q_bert_mask, q_bert_offsets, ocr, ocr_mask, ocr_char, ocr_char_mask, ocr_pos, ocr_position, ocr_ent, ocr_bert, ocr_bert_mask, ocr_bert_offsets, od_offset, ocr_offset, od_last_index, ocr_last_index,od_max_num, ocr_max_num,  ground_truth, ground_truth_YN, ocr_ans_list, file_name, answers, question_id = batch

        # Run forward
        # score_s, score_e: batch x context_word_num
        # score_yes, score_no, score_no_answer: batch x 1
        score_s, score_no_answer, mask, att_score = self.network(img_fea, img_spa, od, od_mask, od_char, od_char_mask, od_pos, od_position, od_ent, od_bert, od_bert_mask, od_bert_offsets, q, q_mask, q_char, q_char_mask, q_pos, q_ent, q_bert, q_bert_mask, q_bert_offsets, ocr, ocr_mask, ocr_char, ocr_char_mask, ocr_pos, ocr_position, ocr_ent, ocr_bert, ocr_bert_mask, ocr_bert_offsets, od_offset, ocr_offset, od_last_index, ocr_last_index,od_max_num, ocr_max_num, return_score=self.reture_att_score)
        score_s.data.masked_fill_(mask.data, -float('inf'))
        scores = torch.cat((score_s, score_no_answer), dim=1)
        targets = torch.cat([ground_truth, ground_truth_YN.unsqueeze(1)], dim=1)
        prob = F.softmax(scores, dim = 1).detach().cpu() # Transfer to CPU/normal tensors for numpy ops
        # prob = F.softmax(targets.float(), dim = 1).data.cpu()

        # Get argmax text spans
        predictions = []
        confidence = []
        
        pred_json = []
        pred_idx = []
        batch_size = scores.size(0)
        context_len = score_s.size(1)
        # print(prob.size())
        for i in range(batch_size):
            _, ids = torch.sort(prob[i, :], descending=True)
            idx = 0
            best_id = ids[idx]
            if best_id == context_len and len(ocr_ans_list[i]) > 0:
                best_id = ids[1]
            # best_id = ids[6]
            pred_idx.append(best_id)

            confidence.append(float(prob[i, best_id]))
            if best_id < len(ocr_ans_list[i]):
                predictions.append(ocr_ans_list[i][best_id])
            elif best_id == context_len:
                predictions.append('')
            else:
                predictions.append('')
                # print('#####ERROR pred_idx')
                # print('_: ', _.data)
                # print('ids: ', ids.data)
                # print('mask: ', mask[i].data)
                # print('scores: ', scores[i].data)
                # print('targets: ', targets[i].data)
                # print('ans_list: ', ocr_ans_list[i])
                # print('prob: ', prob[i].data)
                # assert False
            all_prediction = []
            if all_ans:
                l = ocr_offset[i][1] - ocr_offset[i][0]
                for j in range(30):
                    _id = ids[j]
                    conf = float(prob[i, _id])
                    # print('list: {0:3d} _id: {1:3d} conf: {2:4f} no_ans: {3:3d}'.format(len(ocr_ans_list[i]), _id, conf, context_len))
                    if _id == context_len:
                        pred_word = ''
                    else:
                        pred_word = ocr_ans_list[i][_id]
                    all_prediction.append({'conf':conf, 'pred_word':pred_word})
            _att_score = {}
            if self.reture_att_score:
                for k, v in att_score.items():
                    if 'deep_att_' in k:
                        _att_score[k] = [v[_j][i].tolist() for _j in range(len(v))]
                    else:
                        _att_score[k] = v[i].tolist()
            pred_json.append({
                'file_name': file_name[i],
                'answer': predictions[-1],
                'ground_truth': answers[i],
                'pred_idx': best_id.detach().cpu().item(),
                'answer_pool_len': (scores.size(1) - torch.sum(mask[i])).detach().cpu().item(),
                'no_answer_idx': context_len,
                'question_id': question_id[i],
                'all_prediction': all_prediction,
                'att_score': _att_score,
                'prediction_score': prob[i][best_id].item(),
            })
        assert len(predictions) == batch_size
        # torch.cuda.empty_cache()
        return (predictions, confidence, pred_json) # list of strings, list of floats, list of jsons

    def load_model(self, model_path):
        print('Loading model from', model_path)
        checkpoint = torch.load(model_path)
        state_dict = checkpoint['state_dict']
        new_state = set(self.network.state_dict().keys())
        for k in list(state_dict['network'].keys()):
            if k not in new_state:
                del state_dict['network'][k]
        for k, v in list(self.network.state_dict().items()):
            if k not in state_dict['network']:
                state_dict['network'][k] = v
        self.network.load_state_dict(state_dict['network'])

        print('Loading finished', model_path)        

    def save(self, filename, epoch, prev_filename):
        params = {
            'state_dict': {
                'network': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'updates': self.updates # how many updates
            },
            'train_loss': {
                'val': self.train_loss.val,
                'avg': self.train_loss.avg,
                'sum': self.train_loss.sum,
                'count': self.train_loss.count
            },
            'config': self.opt,
            'epoch': epoch
        }
        try:
            torch.save(params, filename)
            self.log('model saved to {}'.format(filename))
            if os.path.exists(prev_filename):
                os.remove(prev_filename)
        except BaseException:
            self.log('[ WARN: Saving failed... continuing anyway. ]')

    def save_for_predict(self, filename, epoch):
        network_state = dict([(k, v) for k, v in self.network.state_dict().items() if k[0:4] != 'CoVe' and k[0:4] != 'ELMo' and k[0:9] != 'AllenELMo' and k[0:4] != 'Bert'])

        if 'eval_embed.weight' in network_state:
            del network_state['eval_embed.weight']
        if 'fixed_embedding' in network_state:
            del network_state['fixed_embedding']
        params = {
            'state_dict': {'network': network_state},
            'config': self.opt,
        }
        try:
            torch.save(params, filename)
            self.log('model saved to {}'.format(filename))
        except BaseException:
            self.log('[ WARN: Saving failed... continuing anyway. ]')
    def instance_bce_with_logits(self, logits, labels):
        labels = labels.float()
        assert logits.dim() == 2

        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss *= labels.size(1)
        return loss

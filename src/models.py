import logging
import math
import os
import sys

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss,Softmax
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import *
from transformers import RoCBertModel, RoCBertConfig,RoCBertPreTrainedModel
import numpy as np
import copy
import pickle
import json
import string
class SpellBert(BertPreTrainedModel):
    def __init__(self, config):
        super(SpellBert, self).__init__(config)

        self.vocab_size = config.vocab_size
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.vocab_size)
        self.segment = nn.Linear(config.hidden_size, 1)
        self.detect = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def tie_cls_weight(self):
     

        self.classifier.weight = self.bert.embeddings.word_embeddings.weight

    @staticmethod
    def build_batch(batch, tokenizer):
        return batch
    def get_merge_logits(self,batch,hidden_state,error):
        for i in range(batch['src_idx'].shape[0]):
            hidden_state_ = hidden_state.clone()
            for item in error[i]:
                if type(item[0]) != int :
                    if item[0][1]+1 >126:
                        break
                    # import pdb
                    # pdb.set_trace()
                    hidden_state_[i][item[0][0]+1]= hidden_state[i][item[0][0]+1:item[0][1]+2].mean(dim=0)
        return hidden_state_
    def forward(self, batch):
        input_ids = batch['src_idx']
        label_ids = batch['tgt_idx'] if 'tgt_idx' in batch else None
        segment_ids = batch['segment_idx']
        detect_ids = batch['detect']
        attention_mask = batch['masks']
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        error = batch['pinyin_error']
        sequence_output = outputs[0]
        if label_ids is not None:
             sequence_output = self.get_merge_logits(batch, sequence_output,error)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        segment_out = nn.Sigmoid()(self.segment(sequence_output))
        detect_out = self.detect(sequence_output)
        
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if label_ids is not None:
            
            loss_mask = batch['loss_masks']
            loss_fct = CrossEntropyLoss()
            loss_f = nn.BCELoss()
            # Only keep active parts of the loss
        
            active_loss = loss_mask.view(-1) == 1
            active_logits = logits.view(-1, self.vocab_size)[active_loss]
            active_labels = label_ids.view(-1)[active_loss]
            loss_c = loss_fct(active_logits, active_labels)
            loss_seg = loss_f(segment_out.view(-1)[active_loss],segment_ids.float().view(-1)[active_loss])
            loss_de = loss_fct(detect_out.view(-1,28)[active_loss],detect_ids.view(-1)[active_loss])
            #loss = 0.5*loss_de+0.5*loss_c
            loss = 0.3*loss_de+0.3*loss_seg+0.4*loss_c
            #loss = loss_c
            outputs = (loss,) + outputs

        return outputs,segment_out,detect_out

class RocBert(RoCBertPreTrainedModel):
    def __init__(self, config):
        super(RocBert, self).__init__(config)
        
        self.vocab_size = config.vocab_size
        self.bert = RoCBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.vocab_size)
        self.segment = nn.Linear(config.hidden_size, 1)
        self.detect = nn.Linear(config.hidden_size, 28)
        self.fc_detect =  nn.Linear(config.hidden_size,config.hidden_size)
        self.init_weights()
        # with open(config.pinyin_dict_path, 'r') as f:
        #     self.pinyin_dict = json.load(f)
        
    def tie_cls_weight(self):
        
        self.classifier.weight = self.bert.embeddings.word_embeddings.weight
    def get_decoupled_loss(self,batch,logits,labels,active_loss,error):
        with open('/opt/data/private/AlipaySEQ-main/bert/pretrained/pinyin_dict.json', 'r') as f:
            pinyin_dict = json.load(f)
        device = torch.device("cuda")
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.vocab_size)[active_loss], labels.view(-1)[active_loss])
        all_loss_a = []
        all_loss_b = []
        for i in range(batch['src_idx'].shape[0]):
            for item in error[i]:
                if type(item[0]) == int:
                    if item[0]>126 or batch['tgt_idx'][i][item[0]+1] ==100: continue
                    new_vocab = pinyin_dict[batch['tgt_idx'][i][item[0]+1]]
                    
                    indices = torch.LongTensor(new_vocab)
                    #label = new_vocab.index(int(batch['tgt_idx'][i][item[0]+1]))
                    #label = torch.tensor(label,device=device)
                  
                    label=batch['tgt_idx'][i][item[0]+1]
                    logits_a = logits.clone()
                    logits_b = logits.clone()
                    logits_b[i][item[0]+1][indices] = -torch.inf 
                    logits_a[i][item[0]+1][~torch.isin(torch.arange(self.vocab_size), indices)] = -torch.inf 
                    f = Softmax(dim=-1)
                 
                    loss_a = loss_fct(logits_a[i][item[0]+1].view(1,-1),label.view(-1))
                    #loss_b = -loss_fct(logits_b[i][item[0]+1].view(1,-1),label.view(-1))
                   
                    #loss_b = f(logits_b[i][item[0]+1]).mean()
                    all_loss_a.append(loss_a)
                    #all_loss_b.append(loss_b)        
                else:
                    new_vocab = []
                    for j in range(item[0][0],item[0][1]+1):
                        new_vocab += pinyin_dict[batch['src'][i][j]]
                    new_vocab = list(set(new_vocab))
                    indices = torch.LongTensor(new_vocab)
                    loss_1 = 0
                    loss_2 = 0
                    for j in range(item[0][0],item[0][1]+1):
                        if j>126 or batch['tgt_idx'][i][j+1]==100: continue
                        
                        # label = new_vocab.index(int(batch['tgt_idx'][i][j+1]))
                        # label = torch.tensor(label,device=device)
                        # logits_a = logits[i][j+1][indices]   
                        # logits_b = logits[i][j+1][torch.logical_not(torch.isin(torch.arange(self.vocab_size), indices))]
                        
                        # loss_a = loss_fct(logits_a.view(1,-1),label.view(-1))
                        # #loss_b = -torch.log_softmax(logits_b, dim=-1).mean()
                        f = Softmax(dim=-1)
                        #loss_b = f(logits_b).mean()
                        label=batch['tgt_idx'][i][j+1]
                        logits_a = logits.clone()
                        logits_b = logits.clone()
                        logits_b[i][j+1][indices] = -torch.inf 
                        logits_a[i][j+1][~torch.isin(torch.arange(self.vocab_size), indices)] = -torch.inf 
                        loss_a = loss_fct(logits_a[i][j+1].view(1,-1),label.view(-1))
                        #loss_b = -loss_fct(logits_b[i][j+1].view(1,-1),label.view(-1))
                        #loss_b = f(logits_b[i][j+1]).mean()
                        loss_1 += loss_a
                        #loss_2 += loss_b
                    all_loss_a.append(loss_1)
                    #all_loss_b.append(loss_2)
        total_a=torch.tensor(all_loss_a).mean()
        #total_b = torch.tensor(all_loss_b).mean()
        total_loss = loss+total_a
        return total_loss,total_a,loss
    def get_merge_logits(self,batch,hidden_state,error):
        for i in range(batch['src_idx'].shape[0]):
            hidden_state_ = hidden_state.clone()
            for item in error[i]:
                if type(item[0]) != int :
                    if item[0][1]+1 >126:
                        break
                    # import pdb
                    # pdb.set_trace()
                    hidden_state_[i][item[0][0]+1]= hidden_state[i][item[0][0]+1:item[0][1]+2].mean(dim=0)
        return hidden_state_
   
    @staticmethod
    def build_batch(batch, tokenizer):
        return batch
    
        
    def forward(self, batch):
        input_ids = batch['src_idx']
        label_ids = batch['tgt_idx'] if 'tgt_idx' in batch else None
        segment_ids = batch['segment_idx']
        detect_ids = batch['detect']
        attention_mask = batch['masks']
        error = batch['pinyin_error']
        
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        
        sequence_output = outputs[0]
        
        sequence_output = self.dropout(sequence_output)
        segment_out = nn.Sigmoid()(self.segment(sequence_output))
        detect_out = self.detect(sequence_output)
        # if label_ids is not None:
        #     sequence_output = self.get_merge_logits(batch, sequence_output,error)
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if label_ids is not None:
            new_labels = batch['new_tgt']
            loss_mask = batch['loss_masks']
            loss_fct = CrossEntropyLoss()
            loss_f = nn.BCELoss()
            # Only keep active parts of the loss
            
            active_loss = loss_mask.view(-1) == 1
            active_logits = logits.view(-1, self.vocab_size)[active_loss]
            active_labels = label_ids.view(-1)[active_loss]
           
            #decoupled_loss,a,b,c=self.get_decoupled_loss(batch,logits,new_labels,active_loss,error)
            loss_c = loss_fct(active_logits, active_labels)
            # import pdb
            # pdb.set_trace()
            loss_seg = loss_f(segment_out.view(-1)[active_loss],segment_ids.float().view(-1)[active_loss])
            loss_de = loss_fct(detect_out.view(-1,28)[active_loss],detect_ids.view(-1)[active_loss])
            #loss = 0.5*loss_de+0.5*loss_c
            #loss = 0.3*loss_de+0.3*loss_seg+0.4*loss_c
            loss = loss_c
            outputs = (loss,) + outputs

        return outputs,segment_out,detect_out
class RocBert_new(RoCBertPreTrainedModel):
    def __init__(self, config,path):
        super(RocBert_new, self).__init__(config)
        self.vocab_size = config.vocab_size
        self.bert = RoCBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.vocab_size)
        self.segment = nn.Linear(config.hidden_size, 1)
        self.detect = nn.Linear(config.hidden_size, 28)
        self.fc_detect =  nn.Linear(config.hidden_size,config.hidden_size)
        self.init_weights()
        device = torch.device("cuda")
        with open(path, 'r') as f:
            self.pinyin_dict = json.load(f)
        self.sub_cls = [] 
        self.sub_fc = []   
        for v in self.pinyin_dict.values():
            self.sub_fc.append(nn.Linear(config.hidden_size,config.hidden_size).to(device))
            self.sub_cls.append(nn.Linear(config.hidden_size,len(v)).to(device))
            

            
            
    def tie_cls_weight(self):
        
        self.classifier.weight = self.bert.embeddings.word_embeddings.weight
    def get_decoupled_loss(self,batch,hidden_state,error):
       
        device = torch.device("cuda")
        loss_fct = CrossEntropyLoss()
        letters = list(string.ascii_lowercase)
        all_loss = []
    
        for i in range(batch['src_idx'].shape[0]):
            for item in error[i]:
                if type(item[0]) == int:
                    if item[0]>126 or batch['tgt_idx'][i][item[0]+1] ==100: continue
                    new_vocab = self.pinyin_dict[letters[batch['detect'][i][item[0]+1]-2]]
                    
                    hidden_state_sub = self.sub_fc[batch['detect'][i][item[0]+1]-2](hidden_state[i][item[0]+1])
                    logits = self.sub_cls[batch['detect'][i][item[0]+1]-2](hidden_state_sub)
                    label = new_vocab.index(int(batch['tgt_idx'][i][item[0]+1]))
                    label = torch.tensor(label).to(device)
                    loss = loss_fct(logits.view(1,-1),label.view(-1))
                    #loss_b = -loss_fct(logits_b[i][item[0]+1].view(1,-1),label.view(-1))
                   
                    #loss_b = f(logits_b[i][item[0]+1]).mean()
                    all_loss.append(loss)
                    #all_loss_b.append(loss_b)        
                else:
                    
                    loss_1 = 0
                   
                    for j in range(item[0][0],item[0][1]+1):
                        if j>126 or batch['tgt_idx'][i][j+1]==100: continue
                     
                    
                        new_vocab = self.pinyin_dict[letters[batch['detect'][i][j+1]-2]]
                        hidden_state_sub = self.sub_fc[batch['detect'][i][j+1]-2](hidden_state[i][j+1])
                        logits = self.sub_cls[batch['detect'][i][j+1]-2](hidden_state_sub)
                        label = new_vocab.index(int(batch['tgt_idx'][i][j+1]))
                        label = torch.tensor(label).to(device)
                        loss = loss_fct(logits.view(1,-1),label.view(-1))
                            #loss_b = -loss_fct(logits_b[i][j+1].view(1,-1),label.view(-1))
                            #loss_b = f(logits_b[i][j+1]).mean()
                        loss_1 += loss
                            #loss_2 += loss_b
                        all_loss.append(loss_1)
                    #all_loss_b.append(loss_2)
        total_loss=torch.tensor(all_loss).mean()
      
        return total_loss
    def get_merge_logits(self,batch,logits,error):
        for i in range(batch['src_idx'].shape[0]):
            for item in error[i]:
                if type(item[0]) != int :
                    for j in range(item[0][0]+1,item[0][1]+1):
                        if j>126:
                            break
                        logits[i][item[0][0]+1] +=  logits[i][j+1]
        return logits
   
    @staticmethod
    def build_batch(batch, tokenizer):
        return batch
    
        
    def forward(self, batch):
        input_ids = batch['src_idx']
        label_ids = batch['tgt_idx'] if 'tgt_idx' in batch else None
        segment_ids = batch['segment_idx']
        detect_ids = batch['detect']
        attention_mask = batch['masks']
        error = batch['pinyin_error']
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        
        sequence_output = outputs[0]
        
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        #logits = self.get_merge_logits(batch,logits,error)
        segment_out = nn.Sigmoid()(self.segment(sequence_output))
        detect_out = self.detect(sequence_output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if label_ids is not None:
            new_labels = batch['new_tgt']
            loss_mask = batch['loss_masks']
            loss_fct = CrossEntropyLoss()
            loss_f = nn.BCELoss()
            # Only keep active parts of the loss
            
            active_loss = loss_mask.view(-1) == 1
            active_logits = logits.view(-1, self.vocab_size)[active_loss]
            active_labels = new_labels.view(-1)[active_loss]
           
           
            loss_c = loss_fct(active_logits, active_labels)
            decoupled_loss=self.get_decoupled_loss(batch,sequence_output,error)
            loss_c +=decoupled_loss
            # import pdb
            # pdb.set_trace()
            loss_seg = loss_f(segment_out.view(-1)[active_loss],segment_ids.float().view(-1)[active_loss])
            loss_de = loss_fct(detect_out.view(-1,28)[active_loss],detect_ids.view(-1)[active_loss])
            #loss = 0.5*loss_de+0.5*loss_c
            loss = 0.3*loss_de+0.3*loss_seg+0.4*loss_c
            #loss = loss_c
            outputs = (loss,) + outputs

        return outputs,decoupled_loss,segment_out,detect_out,sequence_output
class SRFBert(BertPreTrainedModel):
    def __init__(self, config):
        super(SRFBert, self).__init__(config)

        self.vocab_size = config.vocab_size
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.vocab_size)

        self.init_weights()

    def tie_cls_weight(self):
        self.classifier.weight = self.bert.embeddings.word_embeddings.weight

    @staticmethod
    def build_batch(batch, tokenizer):
        return batch

    def forward(self, batch):
        input_ids = batch['src_idx']
        attention_mask = batch['masks']
        loss_mask = batch['loss_masks']
        label_ids = batch['tgt_idx'] if 'tgt_idx' in batch else None
        bsz, max_length = input_ids.shape

        outputs = self.bert(input_ids, attention_mask=attention_mask)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)

        logits = self.classifier(sequence_output)
        last_hidden_state = outputs.last_hidden_state

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if label_ids is not None:
            # Only keep active parts of the CrossEntropy loss
            loss_fct = CrossEntropyLoss()
            active_loss = loss_mask.view(-1) == 1
            active_logits = logits.view(-1, self.vocab_size)[active_loss]
            active_labels = label_ids.view(-1)[active_loss]
            loss1 = loss_fct(active_logits, active_labels)

            if self.training:

                # Contrastive Probability ptimization Loss
                wrong_judge_positions = ~torch.eq(input_ids, label_ids)

                # self refine
                new_input_ids = copy.deepcopy(input_ids)
                # new_input_ids=copy.deepcopy(label_ids)

                pred_label = active_logits.argmax(-1)

                wrong_judge_positions = ~torch.eq(input_ids, new_input_ids)

                new_input_ids.view(-1)[active_loss] = pred_label
                if new_input_ids.view(-1)[wrong_judge_positions.view(-1)].size(0):
                    new_input_ids.view(-1)[wrong_judge_positions.view(-1)] = logits.view(-1, self.vocab_size)[
                        wrong_judge_positions.view(-1)].argmax(-1)

                # no_mask_position=pred_label!=103
                # new_input_ids.view(-1)[active_loss][no_mask_position]=pred_label[no_mask_position]

                new_outputs = self.bert(new_input_ids, attention_mask=attention_mask)
                #
                new_sequence_output = new_outputs[0]

                new_sequence_output = self.dropout(new_sequence_output)

                new_logits = self.classifier(new_sequence_output)

                loss_fct = CrossEntropyLoss()
                new_active_logits = new_logits.view(-1, self.vocab_size)[active_loss]
                loss3 = loss_fct(new_active_logits, active_labels)
                #
                p = torch.log_softmax(new_active_logits, dim=-1)
                p_tec = torch.softmax(new_active_logits, dim=-1)
                q = torch.log_softmax(active_logits, dim=-1)
                q_tec = torch.softmax(active_logits, dim=-1)

                kl_loss = torch.nn.functional.kl_div(p, q_tec, reduction='none').sum()
                reverse_kl_loss = torch.nn.functional.kl_div(q, p_tec, reduction='none').sum()

                #
                loss = 0.5 * (loss1 + loss3)
                loss += 0.001 * (kl_loss + reverse_kl_loss) / 2

                outputs = (loss,) + outputs


            else:
                outputs = (loss1,) + outputs

                total_logits = logits
                for i in range(2):
                    new_input_ids = copy.deepcopy(input_ids)

                    pred_label = active_logits.argmax(-1)

                    new_input_ids.view(-1)[active_loss] = pred_label

                    new_outputs = self.bert(new_input_ids, attention_mask=attention_mask)

                    new_sequence_output = new_outputs[0]

                    new_sequence_output = self.dropout(new_sequence_output)

                    new_logits = self.classifier(new_sequence_output)

                    total_logits += new_logits

                    input_ids = copy.deepcopy(new_input_ids)
                    active_logits = new_logits.view(-1, self.vocab_size)[active_loss]

                # outputs=(outputs[0],(new_logits+logits)/2)
                outputs = (outputs[0], new_logits)

        return outputs
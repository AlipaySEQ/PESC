import argparse
import os
import pickle
import string

import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizer
import json
import numpy as np
 
np.set_printoptions(threshold=np.inf)

from models import SpellBert, SRFBert,RocBert
import pandas as pd
from metric_core import *
from Chartokenizer import chartokenizer
# from utils import Pinyin2

 
 

# pho2_convertor = Pinyin2()
 


MODEL_CLASSES = {
    'bert': SpellBert,
    'ecopobert': SRFBert,
    'rocbert':RocBert
}



def build_batch(batch):
    # src_idx = batch['src_idx'].flatten().tolist()
    # chars = tokenizer.convert_ids_to_tokens(src_idx)
    # pho_idx, pho_lens = pho2_convertor.convert(chars)
    # batch['pho_idx'] = pho_idx
    # batch['pho_lens'] = pho_lens
    return batch
def get(t):
    if isinstance(t[0],int):
        return t[0]
    else:
        return t[0][0]


def make_features(args,examples, tokenizer, batch_processor):
    max_length = args.max_seq_length
    batch = {}
    
    for t in ['id', 'src', 'tgt', 'tokens_size', 'lengths', 'src_idx', 'masks', 'loss_masks','alpha_idx','error',"segment_idx","detect","pinyin_error","new_tgt"]:
        batch[t] = []
    for item in examples:
        for t in item:
            if t == 'src_idx' :
                seq = item[t][:max_length]
                padding_length = max_length - len(seq)
                batch[t].append(seq + ([0] * padding_length))
                if t == 'src_idx':
                    batch['masks'].append(([1] * len(seq)) + ([0] * padding_length))
            elif t == 'lengths':
                batch[t].append(item[t])
                loss_mask = [0] * max_length
                for i in range(1, min(item[t]+1, max_length)):
                    loss_mask[i] = 1
                batch['loss_masks'].append(loss_mask)
            elif t == "tgt_idx":
                pass
           
            else:
                batch[t].append(item[t])
   
    batch['src_idx'] = torch.tensor(batch['src_idx'], dtype=torch.long)
    
    batch['masks'] = torch.tensor(batch['masks'], dtype=torch.long)
    batch['loss_masks'] = torch.tensor(batch['loss_masks'], dtype=torch.long)
    
    
  
    batch = batch_processor(batch)
    return batch



def prepare_batches(args, test_picke_path, tokenizer_path):
    # tokenizer = BertTokenizer.from_pretrained(tokenizer_path)


    with open(test_picke_path, 'rb') as f:
        dataset = pickle.load(f)


    bs = 128
    batches = []
    r = len(dataset)
    for i in tqdm(range(0, len(dataset), bs)):
        
        batches.append(make_features(
            args,
            dataset[i:min(i + bs, r)],
            None,
            batch_processor=build_batch,
        ))


    return batches

def pinyin_decode(logits,batch,pinyin_dict,detect_out):
    lower = string.ascii_lowercase
    letters = list(lower)
    for i in range(batch['src_idx'].shape[0]):
        for j in range(len(batch['src'][i])):
            if len((batch['src'][i]))<128:
                #a = detect_out[i][1:]
                if int(detect_out[i][j]) != 0 and batch['src'][i][j-1] in pinyin_dict:
                    

                    in_v = pinyin_dict[letters[detect_out[i][j]-2]]
                    not_in_v = set(range(21128)) -set(in_v)
                    for n in not_in_v:
                        logits[i][j][n]=-100
          
    return logits

def merge(batch,hidden_state,segment):
    for i in range(batch['src_idx'].shape[0]):
        hidden_state_ = hidden_state.clone()
        pinyin = []
                
        #hidden_state_[i][item[0][0]+1]= hidden_state[i][item[0][0]+1:item[0][1]+2].mean(dim=0)
        for index in range(batch["lengths"][i]-1,-1,-1):
                if index >127: continue
                if segment[i][index] ==0:
                    pinyin.insert(0,index)
                    continue
                if len(pinyin):
                    hidden_state_[i][pinyin[0]]= hidden_state[i][pinyin[0]:pinyin[-1]].mean(dim=0)
                    pinyin= []
                    continue
    return hidden_state_




def test(ckpt_dir, data_dir, ckpt_num, output_dir, device):

    
    with open('/opt/data/private/zzl/AlipaySEQ-main/bert/roc_pretrained/pinyin_dict.json', 'r') as f:
        pinyin_dict = json.load(f)
    
    tokenizer = chartokenizer.from_pretrained("./pretrained")


    model_dir = os.path.join(ckpt_dir, 'epoch_' + str(ckpt_num))
    weight_dir = model_dir



    test_picke_path = os.path.join(data_dir, 'test.pkl')



    # model_type
    training_args = torch.load(os.path.join(weight_dir, 'training_args.bin'))
    model_type = training_args.model_type


    model_class = MODEL_CLASSES[model_type]


    # Log
    print(f'model_type: {model_type}')
    print(f'weight_dir: {weight_dir}')


    # Dataset
    batches = prepare_batches(
        args=args,
        test_picke_path=test_picke_path,
        tokenizer_path=weight_dir,
    )
    print('test_batches:', len(batches))


    # Device
    device = torch.device(device)
  

    # Model
    print('Load model begin...')
    model = model_class.from_pretrained(model_dir)
    model = model.to(device)
    model = model.eval()
    print('Load model done.')
   
    # Test epoch
    for batch in tqdm(batches):
      
        for t in batch:
            if t not in ['id', 'src', 'tgt', 'lengths', 'tokens_size', 'pho_lens','alpha_idx','error',"segment_idx","detect","pinyin_error","new_tgt"]:
                batch[t] = batch[t].to(device)


        with torch.no_grad():
            outputs , segment_out, detect_out= model(batch)

      
        logits = outputs[0]
        #sequence_output = merge(batch,sequence_output)
        #logits = model.classifier(sequence_output)
        segment_out = segment_out.detach().cpu().numpy()
        segment_out = np.where(segment_out > 0.5, 1, 0)
        detect_out = detect_out.detach().cpu().numpy()
        #detect_out = np.where(detect_out > 0.5, 1, 0)
        detect_out=np.argmax(detect_out, axis=-1)
        #sequence_output = merge(batch,sequence_output,segment_out)
        #logits = model.classifier(sequence_output)
        preds = logits.detach().cpu().numpy()
        import pdb
        pdb.set_trace()
        #preds = pinyin_decode(preds,batch,pinyin_dict,detect_out)
        # print(preds[1][9])
        # f = open("a.txt","w")
        # f.writelines(str(preds[1][9]))
        
        # _,preds = logits.topk(2, dim=-1, largest=True, sorted=True)
        # preds_1 = preds[:,:,0].cpu().numpy()
        # preds_2 = preds[:,:,1].cpu().numpy()
        # import pdb
        # pdb.set_trace()
        preds = np.argmax(preds, axis=-1)
        # import pdb
        # pdb.set_trace()
       
        
       
        batch['src_idx'] = batch['src_idx'].detach().cpu().numpy()
        batch['pred_idx'] = preds
        #batch['next_pred'] = preds_2
        batch["segment_pre"] = segment_out
        batch["detect_pre"] = detect_out

    # Metric
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)


    pred = []
    target = []
    pinyin_pred = []
    pinyin_target = [] 
    f = open("result_2.txt","w",encoding="utf8")
    total = 0
    cor = 0
    tor = 0
    for batch in batches:
        
        for i in range(batch['src_idx'].shape[0]):
            target.append(batch['error'][i])
            
            pred_idx = batch['pred_idx'][i][1:1+batch['lengths'][i]]
            #next_pred = batch['next_pred'][i][1:1+batch['lengths'][i]]
            src_idx = batch['src_idx'][i][1:1+batch['lengths'][i]]
            # import pdb
            # pdb.set_trace()
            segment_pre = batch["segment_pre"][i][1:1+batch['lengths'][i]]
            segment_pre = segment_pre.flatten().tolist() 
            detect_pre = batch["detect_pre"][i][1:1+batch['lengths'][i]]
            detect_pre = detect_pre.flatten().tolist() 
            # import pdb
            # pdb.set_trace()
            alpha_idx = batch['alpha_idx'][i][1:-1]
            
            pinyin_tgt = []
            total += 1
            pinyin = []
            # #分割
            # for index in range(len(pred_idx)-1,-1,-1):
                
            #     # if detect_pre[index] == 1 and pred_idx[index]==1:
                    
                    
            #     #     pred_idx[index] == next_pred[index]

            #     if segment_pre[index] ==0:
            #         pinyin.insert(0,index)
            #         continue
            #     if len(pinyin):
            #         for id in pinyin:
            #             pred_idx[id] = 1
            #         pinyin= []
            #         continue
        
            if segment_pre == batch["segment_idx"][i]:
                 cor += 1
            if detect_pre == batch["detect"][i]:            
                 tor += 1
            for item in batch['error'][i]:
                
                idx = get(item)
                if alpha_idx[idx]==1:
                    pinyin_tgt.append(item)
              
                
            pinyin_target.append(set(pinyin_tgt)) 
            
            pred_point = []            
            pinyin_idx = []
            for j in range(len(src_idx)-1,-1,-1):
                if pred_idx[j] ==1:
                    pinyin_idx.insert(0,j)
                    continue
                if len(pinyin_idx):
                    pred_point.append(((j,pinyin_idx[-1]),tokenizer.convert_ids_to_tokens(int(pred_idx[j]))))
                    pinyin_idx = []
                    continue
                if src_idx[j]!=pred_idx[j]:
                    pred_point.append((j,tokenizer.convert_ids_to_tokens(int(pred_idx[j]))))
            pred.append(set(pred_point))
            # 只计算字母部分
            pred_point = []            
            pinyin_idx = []
            for j in range(len(src_idx)-1,-1,-1):
                if alpha_idx[j]==1:
                    if pred_idx[j] ==1:
                        pinyin_idx.insert(0,j)
                        continue
                    if len(pinyin_idx):
                        pred_point.append(((j,pinyin_idx[-1]),tokenizer.convert_ids_to_tokens(int(pred_idx[j]))))
                        pinyin_idx = []
                        continue
                    if src_idx[j]!=pred_idx[j]:
                        pred_point.append((j,tokenizer.convert_ids_to_tokens(int(pred_idx[j]))))
            pinyin_pred.append(set(pred_point))

            while 1 in pred_idx:
                pred_idx = list(pred_idx)
                pred_idx.remove(1)
            pre = "".join(tokenizer.convert_ids_to_tokens(pred_idx))
           
            # import pdb
            # pdb.set_trace()
            if pre!= batch["tgt"][i]:
                if len(pre)!=len(batch["tgt"][i]):
                        f.writelines(batch['src'][i]+"/"+pre+"/"+batch["tgt"][i]+"/")
                        f.writelines("\n")
                    
        
    print(cor)
    print(total)
    print('分割正确率:',cor/total)
    print('dec正确率:',tor/total)
    # import pdb
    # pdb.set_trace()
    results = sent_metric_correct(pred,target)
    results.update(sent_metric_detect(pred,target))
    # results.update(char_metric_correct(pinyin_pred,pinyin_target))


    return results



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--ckpt_num', type=int, default=-1)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")    
    parser.add_argument('--device', default="cuda: 0")
    args = parser.parse_args()


    result = test(
        ckpt_dir=args.ckpt_dir,
        data_dir=args.data_dir,
        ckpt_num=args.ckpt_num,
        output_dir=args.output_dir,
        device=args.device,
    )


    result = dict((k, v) for k, v in result.items())
    for key in result.keys():
        print(f'{key}: {result[key]:.2f}')
        result[key] = [result[key]]
    result["model"] = [os.path.join(args.ckpt_dir, 'epoch_' + str(args.ckpt_num))]
    results = pd.DataFrame(result)
    result_path = os.path.join(args.output_dir, 'result.csv')
    results.to_csv(result_path, mode='a', index=False, header=None)      
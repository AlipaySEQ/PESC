from __future__ import absolute_import, division, print_function


import argparse
import glob
import logging
import os
import random
import json

from metric_core import *
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler





from tqdm import tqdm, trange
from Chartokenizer import chartokenizer
from transformers import (WEIGHTS_NAME, BertConfig, BertTokenizer,RoCBertConfig,RoCBertModel)


from transformers import AdamW, get_linear_schedule_with_warmup
from models import SpellBert, SRFBert,RocBert,RocBert_new


import pickle


logger = logging.getLogger(__name__)

torch.autograd.set_detect_anomaly(True)
MODEL_CLASSES = {
    'bert': (BertConfig, SpellBert, chartokenizer),
    'ecopobert': (BertConfig, SRFBert, chartokenizer),
    'rocbert':(RoCBertConfig, RocBert, chartokenizer),
    'new':(RoCBertConfig, RocBert_new, chartokenizer)
}



def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



def create_dataset(args, input_file):
    input_file = os.path.join(args.data_dir, input_file)
    dataset = pickle.load(open(input_file, 'rb'))
    
    return dataset

def make_features_new(args,examples, tokenizer, batch_processor):
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
    
    
  
    batch = batch_processor(batch,tokenizer)
    return batch

def make_features(args, examples, tokenizer, batch_processor):
    '''
    max_length = -1
    for item in examples:
        max_length = max(max_length, max(len(item['src_idx']), len(item['tgt_idx'])))
    max_length = min(max_length, args.max_seq_length)
    '''
    max_length = args.max_seq_length
    batch = {}
    for t in ['id', 'src', 'tgt', 'tokens_size', 'lengths', 'src_idx', 'tgt_idx', 'new_tgt','masks', 'loss_masks','alpha_idx','error',"segment_idx","detect",'pinyin_error']:
        batch[t] = []
    for item in examples:
        for t in item:
            if t == 'src_idx' or t == 'tgt_idx' or t =="new_tgt":
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
            elif t == "segment_idx":
                seq = item[t][:max_length-1]
                padding_length = max_length - len(seq) - 1
                batch[t].append([1]+ seq + ([0] * padding_length))
                
            elif t == "detect":
                seq = item[t][:max_length-1]
                padding_length = max_length - len(seq) - 1
                batch[t].append([0]+ seq + ([0] * padding_length))
            else:
                batch[t].append(item[t])
    batch['src_idx'] = torch.tensor(batch['src_idx'], dtype=torch.long)
    batch['tgt_idx'] = torch.tensor(batch['tgt_idx'], dtype=torch.long)
    batch['new_tgt'] = torch.tensor(batch['new_tgt'], dtype=torch.long)
    batch['masks'] = torch.tensor(batch['masks'], dtype=torch.long)
    batch['loss_masks'] = torch.tensor(batch['loss_masks'], dtype=torch.long)
    
    batch['segment_idx'] = torch.tensor(batch['segment_idx'], dtype=torch.long)
    batch['detect'] = torch.tensor(batch['detect'], dtype=torch.long)
    batch = batch_processor(batch, tokenizer)
    return batch



def data_helper(args, dataset, tokenizer, batch_processor, is_eval=False):
    if not is_eval:
        random.shuffle(dataset)
        start_position = 0
        width = args.train_batch_size * 5000
        intervals = []
        while start_position < len(dataset):
            intervals.append((start_position, min(start_position + width, len(dataset))))
            start_position += width
        bs = args.train_batch_size
    else:
        intervals = [(0, len(dataset))]
        bs = 64


    for l, r in intervals:
        batches = []
        for i in range(l, r, bs):
            if is_eval:
                batches.append(make_features_new(args, dataset[i:min(i + bs, r)], tokenizer, batch_processor))
            else:
                batches.append(make_features(args, dataset[i:min(i + bs, r)], tokenizer, batch_processor))
        for batch in batches:
            yield batch
        



def train(args, model, tokenizer, batch_processor):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size
    test_dataset = create_dataset(args, args.predict_file)
    if args.local_rank == -1:
        train_dataset = create_dataset(args, args.train_file)
    else:
        total_dataset = create_dataset(args, args.train_file)
        start_position = 0
        width = torch.distributed.get_world_size()
        train_dataset = []
        while start_position + width <= len(total_dataset):
            train_dataset.append(total_dataset[start_position + args.local_rank])
            start_position += width


    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
                    len(train_dataset) // args.train_batch_size // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataset) // args.train_batch_size // args.gradient_accumulation_steps * args.num_train_epochs


    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    need_optimized_parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in need_optimized_parameters if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in need_optimized_parameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]


    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
            amp.register_half_function(torch, 'einsum')
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)


    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)


    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d",
                len(train_dataset) * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)


    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for epoch in train_iterator:
        # epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(data_helper(args, train_dataset, tokenizer, batch_processor, False)):
          
            model.train()
            for t in batch:
                if t not in ['id', 'src', 'tgt', 'lengths', 'tokens_size', 'pho_lens','alpha_idx','error','pinyin_error']:
                    batch[t] = batch[t].to(args.device)

            
            #loss,_= model(batch)[0]
            outputs ,segment_out, detect_out= model(batch)
            loss = outputs[0]
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps


            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()


            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)


                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs['learning_rate'] = learning_rate_scalar
                    logs['loss'] = loss_scalar
                    logging_loss = tr_loss
                    #logger.info("Step: {}, LR: {}, Loss: {},Lossa: {},Lossb: {},Lossc: {}".format(global_step, logs['learning_rate'], logs['loss'],a,b,c))
                    logger.info("Step: {}, LR: {}, Loss: {}".format(global_step, logs['learning_rate'], logs['loss']))
        
        pred = []
        target = []
        for batch in data_helper(args, test_dataset, tokenizer, batch_processor,True):
            model.eval()
            for t in batch:
                if t not in ['id', 'src', 'tgt', 'tokens_size', 'lengths', 'src_idx', 'masks', 'loss_masks','alpha_idx','error',"segment_idx","detect","pinyin_error","new_tgt"]:
                    batch[t] = batch[t].to(args.device)
            with torch.no_grad():
                outputs ,segment_out, detect_out= model(batch)
            logits = outputs[0]
            preds = logits.detach().cpu().numpy()
            preds = np.argmax(preds, axis=-1)
            batch['src_idx'] = batch['src_idx'].detach().cpu().numpy()
            for i in range(batch['src_idx'].shape[0]):
                target.append(batch['error'][i])
                pred_idx = preds[i][1:1+batch['lengths'][i]]
                src_idx = batch['src_idx'][i][1:1+batch['lengths'][i]]
                pred_point = []   
                for j in range(len(src_idx)-1,-1,-1):
                    if src_idx[j]!=pred_idx[j]:
                        pred_point.append((j,tokenizer.convert_ids_to_tokens(int(pred_idx[j]))))
                pred.append(set(pred_point))
        print(pred)
        print(target)
        result = sent_metric_correct(pred,target)
        result.update(sent_metric_detect(pred,target))    
        result = dict((k, v) for k, v in result.items())
        for key in result.keys():
            print(f'{key}: {result[key]:.2f}')
            
        if args.local_rank in [-1, 0]:
            output_dir = os.path.join(args.output_dir, 'epoch_{}'.format(epoch))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model,
                                                    'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            logger.info("Saving model checkpoint to %s", output_dir)


        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
    return global_step, tr_loss / global_step



# def evaluate(args, model, tokenizer, batch_processor, prefix=""):
#     eval_dataset = create_dataset(args, args.dev_file)
#     args.eval_batch_size = args.per_gpu_eval_batch_size


#     logger.info("***** Running evaluation {} *****".format(prefix))
#     logger.info("  Num examples = %d", len(eval_dataset))
#     logger.info("  Batch size = %d", args.eval_batch_size)


#     eval_loss = 0.0
#     nb_eval_steps = 0


#     batches = []


#     for batch in data_helper(args, eval_dataset, tokenizer, batch_processor, True):
#         model.eval()
#         for t in batch:
#             if t not in ['id', 'src', 'tgt', 'lengths', 'tokens_size', 'pho_lens']:
#                 batch[t] = batch[t].to(args.device)
#         with torch.no_grad():
#             outputs = model(batch)
#             tmp_eval_loss, logits = outputs[:2]
#             eval_loss += tmp_eval_loss.mean().item()
#         nb_eval_steps += 1
#         preds = logits.detach().cpu().numpy()
#         preds = np.argmax(preds, axis=-1)
#         batch['src_idx'] = batch['src_idx'].detach().cpu().numpy()
#         batch['pred_idx'] = preds


#         batches.append(batch)


#     metric = Metric(vocab_path=args.output_dir)
#     pred_txt_path = os.path.join(args.output_dir, prefix, "preds.txt")
#     pred_lbl_path = os.path.join(args.output_dir, prefix, "labels.txt")
#     results = metric.metric(
#         batches=batches,
#         pred_txt_path=pred_txt_path,
#         pred_lbl_path=pred_lbl_path,
#         label_path=os.path.join(args.data_dir, args.dev_label_file)
#     )
#     for key in sorted(results.keys()):
#         logger.info("  %s = %s", key, str(results[key]))
#     return results



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    print(1)
    
    ## Required parameters
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")


    ## Other parameters
    parser.add_argument("--model_name_or_path", default="", type=str)
    parser.add_argument("--pinyin_dict_path", default="", type=str)
    parser.add_argument("--data_dir", default="", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")


    parser.add_argument("--train_file", default="train.pkl", type=str)
    parser.add_argument("--dev_file", default="dev.pkl", type=str)
    parser.add_argument("--dev_label_file", default="dev.lbl.tsv", type=str)
    parser.add_argument("--predict_file", default="test.sighan15.pkl", type=str)
    parser.add_argument("--predict_label_file", default="test.sighan15.lbl.tsv", type=str)


    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action='store_true')
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")


    parser.add_argument("--order_metric", default='avg_loss', type=str)
    parser.add_argument("--metric_reverse", action='store_true')
    parser.add_argument("--num_save_ckpts", default=5, type=int)
    parser.add_argument("--remove_unused_ckpts", action='store_true')


    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")


    parser.add_argument('--logging_steps', type=int, default=10,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=1000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")


    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")


    args = parser.parse_args()


    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
    # args.local_rank = int(os.environ["LOCAL_RANK"])
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        #device = torch.device("cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device


    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)


    set_seed(args)


    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    
    model = model_class(config)
    model.bert = RoCBertModel.from_pretrained(args.model_name_or_path, config=config,
                                   cache_dir=args.cache_dir if args.cache_dir else None)
    # model = model_class.from_pretrained(args.model_name_or_path, config=config,
    #                               cache_dir=args.cache_dir if args.cache_dir else None)
    model.tie_cls_weight()


    batch_processor = model_class.build_batch
   
    model.to(args.device)

    print(2)

    logger.info("Training/evaluation parameters %s", args)


    # Training
    if args.do_train:
        global_step, tr_loss = train(args, model, tokenizer, batch_processor)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
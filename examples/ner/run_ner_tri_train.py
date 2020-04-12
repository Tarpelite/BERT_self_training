# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Fine-tuning the library models for named entity recognition on CoNLL-2003 (Bert or Roberta). """


import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
from seqeval.metrics import f1_score, precision_score, recall_score
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from torch.nn.functional import softmax

from transformers import (
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from utils_ner import convert_examples_to_features, get_labels
from utils_ner import read_examples_from_ner_file as read_examples_from_file
from utils_ner import read_examples_from_file as read_tri_examples_from_file
import pickle


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in MODEL_CONFIG_CLASSES), ())

TOKENIZER_ARGS = ["do_lower_case", "strip_accents", "keep_accents", "use_fast"]


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def load_train_data(args, tokenizer, labels, pad_token_label_id):
    # load source data
    examples = read_examples_from_file(args.data_dir, "train")
    features = convert_examples_to_features(
        examples,
        labels,
        args.max_seq_length,
        tokenizer,
        cls_token_at_end=bool(args.model_type in ["xlnet"]),
        # xlnet has a cls token at the end
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=bool(args.model_type in ["roberta"]),
        # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
        pad_on_left=bool(args.model_type in ["xlnet"]),
        # pad on the left for xlnet
        pad_token=tokenizer.pad_token_id,
        pad_token_segment_id=tokenizer.pad_token_type_id,
        pad_token_label_id=pad_token_label_id,
    )

    source_data = []
    for f in features:
        record = [f.input_ids, f.input_mask, f.segment_ids, f.label_ids]
        source_data.append(record)
    
    # load target_data
    examples = read_tri_examples_from_file(args.tgt_file, "test")
    features = convert_examples_to_features(
        examples,
        labels,
        args.max_seq_length,
        tokenizer,
        cls_token_at_end=bool(args.model_type in ["xlnet"]),
        # xlnet has a cls token at the end
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=bool(args.model_type in ["roberta"]),
        # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
        pad_on_left=bool(args.model_type in ["xlnet"]),
        # pad on the left for xlnet
        pad_token=tokenizer.pad_token_id,
        pad_token_segment_id=tokenizer.pad_token_type_id,
        pad_token_label_id=pad_token_label_id,
    )

    target_data = []
    for f in features:
        record = [f.input_ids, f.input_mask, f.segment_ids]
        target_data.append(record)

    return source_data, target_data


def labelling(args, all_target_data, model_f1, model_f2, N_init):

    # make target_data_loader
    np.random.shuffle(all_target_data)
    cand_data = all_target_data[:N_init]
    all_input_ids = torch.tensor([x[0] for x in cand_data], dtype=torch.long)
    all_input_mask = torch.tensor([x[1] for x in cand_data], dtype=torch.long)
    all_segment_ids = torch.tensor([x[2] for x in cand_data], dtype=torch.long)

    dataset = TensorDataset(all_input_ids,
    all_input_mask, all_segment_ids)

    eval_sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.mini_batch_size)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    labeled_data = []
    logger.info("***** Running Labelling*****")

    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model_f1.eval()
    model_f2.eval()
    all_logits1 = []
    all_logits2 = []
    choose_index = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask":batch[1],
                "token_type_ids":batch[2]
            }

            outputs1 = model_f1(**inputs)
            outputs2 = model_f2(**inputs)

            logits1 = outputs1[0] # [batch_size, seq_len, num_labels]
            logits2 = outputs2[0]

            logits1 = softmax(logits1, dim=2)
            logits2 = softmax(logits2, dim=2)

        np.append(all_logits1, logits1.detach().cpu().numpy(), axis=0)
        np.append(all_logits2, logits2.detach().cpu().numpy(), axis=0)

    
    # do collect
    all_preds_max_1 = np.max(all_logits1, axis=2)
    # [batch_size, seq_len, 1]
    all_preds_max_2 = np.max(all_logits2, axis=2)
    # [batch_size, seq_len, 1]

    all_labels_1 = np.argmax(all_logits1, axis=2)
    all_labels_2 = np.argmax(all_logits2, axis=2)
    

    assert len(dataset) == len(all_preds_max_1) == len(all_preds_max_2) == len(all_labels_1) == len(all_labels_2)

    labeled_data = []
    for i in range(len(dataset)):
        record = cand_data[i]

        max_1 = all_preds_max_1[i]
        max_2 = all_preds_max_2[i]

        labels_1 = all_labels_1[i]
        labels_2 = all_labels_2[i]
        Flag = True
        for j in range(len(max_1)):
            if labels_1[j] != labels_2[j]:
                flag = False
                break
            elif max(labels_1[j], labels_2[j]) < args.threshold:
                flag = False
                break
        if not flag:
            continue

        record.append(labels_1)
        labeled_data.append(record)
    
    logger.info("**** collect labeled data size %s", len(labeled_data))

    return labeled_data


def prepare_dataset(source_data, labeled_data):
    ata_L = np.append(source_data, labeled_data, axis=0)
    data_L_input_ids = torch.tensor([x[0] for x in data_L], dtype=torch.long)
    data_L_input_mask = torch.tensor([x[1] for x in data_L], dtype=torch.long)
    data_L_segment_ids = torch.tensor([x[2] for x in data_L], dtype=torch.long)
    data_L_labels_ids = torch.tensor([x[3] for x in data_L], dtype=torch.long)

    dataset_L = TensorDataset(data_L_input_ids,
    data_L_input_mask, data_L_segment_ids, data_L_labels_ids)


    data_S_input_ids = torch.tensor([x[0] for x in data_S], dtype=torch.long)
    data_S_input_mask = torch.tensor([x[1] for x in data_S], dtype=torch.long)
    data_S_segment_ids = torch.tensor([x[2] for x in data_S_segment_ids], dtype=torch.long)
    data_S_labels_ids = torch.tensor([x[3] for x in data_S], dtype=torch.long)

    dataset_S = TensorDataset(data_S_input_ids, data_S_input_mask, data_S_segment_ids, data_S_labels_ids)

    data_TL = labeled_data

    data_TL_input_ids = torch.tensor([x[0] for x in data_TL], dtype=torch.long)
    data_TL_input_mask = torch.tensor([x[1] for x in data_TL], dtype=torch.long)
    data_TL_segment_ids = torch.tensor([x[2] for x in data_TL], dtype=torch.long)
    data_TL_labels_ids = torch.tensor([x[3] for x in data_TL], dtype=torch.long)

    dataset_TL =  TensorDataset(data_TL_input_ids, data_TL_input_mask, data_TL_segment_ids, data_TL_labels_ids)

    return dataset_L, dataset_S, dataset_TL


def tri_train(args, model_f1, model_f2, model_ft, source_data, target_data):

    Nt = args.N_init
    labeled_data = labelling(args, target_data, model_f1, model_f2, Nt)

    dataset_L, dataset_S, dataset_TL = prepare_dataset(source_data, labeled_data)

    k_step = args.k_step
    
    k_iterator = trange(k_step, desc="k_iter", disable=args.local_rank not in[-1, 0])


    for k in k_iterator:
        epoch_iterator = tqdm(args.iter, desc="Iter")
        for _ in epoch_iterator:
            model_f1, model_f2 = train_f1_f2(args, model_f1, model_f2, dataset_L)
            model_ft = train_ft(args,model_ft, dataset_TL)
        
        Nt = int(k/20*len(target_data))
        labeled_data = labelling(args, target_data, model_f1, model_f2, Nt)
        dataset_L, dataset_S, dataset_TL = prepare_dataset(source_data, labeled_data)
    
    return model_f1, model_f2, model_ft



def train_f1_f2(args, model_f1, model_f2, train_dataset):
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()
    
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    args.num_train_epochs = 1
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    if args.warmup_ratio > 0:
        args.warmup_steps = int(t_total * args.warmup_ratio)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in list(model_f1.named_parameters()) + list(model_f2.named_parameters()) if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in list(model_f1.named_parameters()) + list(model_f2.named_parameters()) if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ] 

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

   
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        [model_f1, model_f2], optimizer = amp.initialize([model_f1, model_f2], optimizer, opt_level=args.fp16_opt_level)
    
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model_f1 = torch.nn.DataParallel(model_f1)
        model_f2 = torch.nn.DataParallel(model_f2)

    
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model_f1 = torch.nn.parallel.DistributedDataParallel(
            model_f1, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

        model_f2 = torch.nn.parallel.DistributedDataParallel(
            model_f2, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )
    
 

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
   
    tr_loss, logging_loss = 0.0, 0.0
    model_f1.zero_grad()
    model_f2.zero_grad()


    set_seed(args)

    for _ in range(1):
        epoch_iterator = tqdm(train_dataloader, desc="Iter(loss=X.XXX, lr=X.XXXXXXXX)", disable=args.local_rank not in [-1, 0])

        for step, batch in enumerate(epoch_iterator):
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
                
            model_f1.train()
            model_f2.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "input_ids":batch[0],
                "attention_mask":batch[1],
                "token_type_ids":batch[2],
                "labels":batch[3]
            }

            outputs1 = model_f1(**inputs)
            loss1, _ , s1 = outputs1[:3]

            
            outputs2 = model_f2(**inputs)
            loss2, _,  s2 = outputs2[:3]

            w1 = model_f1.classifier.weight #[hidden_size, num_labels]
            w2 = model_f2.classifier.weight.transpose(-1, -2) #[num_labels, hidden_size]

            norm_term = torch.norm(torch.matmul(w1, w2))

            loss = loss1 + loss2 + args.alpha * norm_term

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                epoch_iterator.set_description('Iter (loss=%5.3f) lr=%9.7f' % (loss.item(), scheduler.get_lr()[0]))
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model_f1.parameters(), args.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(model_f2.parameters(), args.max_grad_norm)
                    

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model_f1.zero_grad()
                model_f2.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                  
                    tb_writer.add_scalar("f1_f2_lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("f1_f2_loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                    

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return model_f1, model_f2


def train_ft(args, model_ft, train_dataset):
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()
    
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    args.num_train_epochs = 1
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    
    if args.warmup_ratio > 0:
        args.warmup_steps = int(t_total * args.warmup_ratio)
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in list(model_ft.named_parameters()) if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in list(model_ft.named_parameters())  if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ] 

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
    
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model_ft, optimizer = amp.initialize(model_ft, optimizer, opt_level=args.fp16_opt_level)
    
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model_ft = torch.nn.DataParallel(model_ft)
    
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model_ft = torch.nn.parallel.DistributedDataParallel(
            model_ft, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )
    

    # Train ft

    # logger.info("*****  Running training ft ******")
    # logger.info("  Num examples = %d", len(train_dataset))
    # logger.info("  Num Epochs = %d", args.num_train_epochs)
    # logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    # logger.info(
    #     "  Total train batch size (w. parallel, distributed & accumulation) = %d",
    #     args.train_batch_size
    #     * args.gradient_accumulation_steps
    #     * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    # )
    # logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    # logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    tr_loss, logging_loss = 0.0, 0.0

    model_ft.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )

    set_seed(args)

    for _ in range(1):
        epoch_iterator = tqdm(train_dataloader, desc="Iter(loss=X.XXX, lr=X.XXXXXXXX)", disable=args.local_rank not in [-1, 0])

        for step, batch in enumerate(epoch_iterator):
            model_ft.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "input_ids":batch[0],
                "attention_mask":batch[1],
                "token_type_ids":batch[2],
                "labels":batch[3]
            }

            outputs = model(**inputs)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu 
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                epoch_iterator.set_description('Iter (loss=%5.3f) lr=%9.7f' % (loss.item(), scheduler.get_lr()[0]))
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model_ft.zero_grad()
                global_step += 1

            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev")
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("ft_lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("ft_loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                    

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return model_ft




def evaluate(args, model, tokenizer, labels, pad_token_label_id, mode, prefix=""):
    eval_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode=mode)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet"] else None
                )  # XLM and RoBERTa don"t use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            if args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)

    label_map = {i: label for i, label in enumerate(labels)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])
    
    for i in range(len(preds_list)):
        for j in range(len(preds_list[i])):
            preds_list[i][j] = preds_list[i][j].split("-")[-1]
    
    for i in range(len(out_label_list)):
        for j in range(len(out_label_list[i])):
            out_label_list[i][j] = out_label_list[i][j].split("-")[-1]


    results = {
        "loss": eval_loss,
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }

    logger.info("***** Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    return results, preds_list


def load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}".format(
            mode, list(filter(None, args.model_name_or_path.split("/"))).pop(), str(args.max_seq_length)
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        examples = read_examples_from_file(args.data_dir, mode)
        features = convert_examples_to_features(
            examples,
            labels,
            args.max_seq_length,
            tokenizer,
            cls_token_at_end=bool(args.model_type in ["xlnet"]),
            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(args.model_type in ["roberta"]),
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(args.model_type in ["xlnet"]),
            # pad on the left for xlnet
            pad_token=tokenizer.pad_token_id,
            pad_token_segment_id=tokenizer.pad_token_type_id,
            pad_token_label_id=pad_token_label_id,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_TYPES),
    )
    parser.add_argument(
        "--model_f1_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--model_f2_path",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model_ft_path",
        default=None,
        type=str,
        required=True
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    
    parser.add_argument(
        "--tgt_file",
        type=str,
        default=None,
    )

    # Other parameters
    parser.add_argument(
        "--labels",
        default="",
        type=str,
        help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Whether to run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )
    parser.add_argument(
        "--keep_accents", action="store_const", const=True, help="Set this flag if model is trained with accents."
    )
    parser.add_argument(
        "--strip_accents", action="store_const", const=True, help="Set this flag if model is trained without accents."
    )
    parser.add_argument("--use_fast", action="store_const", const=True, help="Set this flag to use fast tokenization.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("--N_init", type=int, default=100, help = "the init size of target set")
    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare CONLL-2003 task
    labels = get_labels(args.labels)
    num_labels = len(labels)
    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        id2label={str(i): label for i, label in enumerate(labels)},
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer_args = {k: v for k, v in vars(args).items() if v is not None and k in TOKENIZER_ARGS}
    logger.info("Tokenizer arguments: %s", tokenizer_args)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
        **tokenizer_args,
    )
    # model = AutoModelForTokenClassification.from_pretrained(
    #     args.model_name_or_path,
    #     from_tf=bool(".ckpt" in args.model_name_or_path),
    #     config=config,
    #     cache_dir=args.cache_dir if args.cache_dir else None,
    # )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab


    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        source_data, target_data = load_train_data(args, tokenizer, labels, pad_token_label_id)
        model_f1 = AutoTokenizer.from_pretrained(
            args.model_f1_path,
            from_tf=bool(".ckpt" in args.model_f1_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,

        )

        model_f2 = AutoTokenizer.from_pretrained(
            args.model_f2_path,
            from_tf=bool(".ckpt" in args.model_f1_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

        model_ft = AutoTokenizer.from_pretrained(
            args.model_ft_path,
            from_tf=bool(".ckpt" in args.model_f1_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        model_f1.to(args.device)
        model_f2.to(args.device)
        model_ft.to(args.device)

        model_f1, model_f2, model_ft = tri_train(args, model_f1, model_f2, model_ft, source_data, target_data)
        # train_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="train")
        # global_step, tr_loss = train(args, train_dataset, model, tokenizer, labels, pad_token_label_id)
        # logger.info(" global_step = %s, average loss = %s", global_step, tr_loss) 
        model = model_ft

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir, **tokenizer_args)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = AutoModelForTokenClassification.from_pretrained(checkpoint)
            model.to(args.device)
            result, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev", prefix=global_step)
            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
            results.update(result)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))

    if args.do_predict and args.local_rank in [-1, 0]:
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir, **tokenizer_args)
        model = AutoModelForTokenClassification.from_pretrained(args.output_dir)
        model.to(args.device)
        result, predictions = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="test")
        # Save results
        output_test_results_file = os.path.join(args.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))
        # Save predictions
        output_test_predictions_file = os.path.join(args.output_dir, "test_predictions.txt")
        with open(output_test_predictions_file, "w") as writer:
            with open(os.path.join(args.data_dir, "test.txt"), "r") as f:
                example_id = 0
                for line in f:
                    if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                        writer.write(line)
                        if not predictions[example_id]:
                            example_id += 1
                    elif predictions[example_id]:
                        output_line = line.split()[0] + " " + predictions[example_id].pop(0) + "\n"
                        writer.write(output_line)
                    else:
                        logger.warning("Maximum sequence length exceeded: No prediction for '%s'.", line.split()[0])

    return results


if __name__ == "__main__":
    main()

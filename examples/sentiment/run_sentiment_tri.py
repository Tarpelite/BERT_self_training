import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
from sklearn.model_selection import KFold
from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score
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
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    BertForDQD,
)
from utils_sentiment import *

from sklearn.metrics import roc_auc_score


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

# import pudb
# pudb.set_trace()


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

def labelling(args, all_target_data, model_f1, model_f2, N_init):

    np.random.shuffle(all_target_data)
    cand_data = all_target_data[:N_init]
    all_input_ids = torch.tensor([x.input_ids for x in cand_data], dtype=torch.long)
    all_input_mask = torch.tensor([x.input_mask for x in cand_data], dtype=torch.long)
    all_segment_ids = torch.tensor([x.segment_ids for x in cand_data], dtype=torch.long)
    all_label_ids = torch.tensor([x.label_ids for x in cand_data], dtype=torch.long)

    dataset = TensorDataset(all_input_ids,
    all_input_mask, all_segment_ids, all_label_ids)

    eval_sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.mini_batch_size)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    labeled_data = []
    logger.info("****** Running Labelling ******")

    logger.info(" Num examples = %d", len(dataset))
    logger.info(" Batch size = %d", args.mini_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model_f1.eval()
    model_f2.eval()
    all_logits1 = []
    all_logits2 = []
    choose_index = []
    all_true_labels = []
    all_pseudo_labels = []

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids":batch[0],
                "attention_mask":batch[1],
                "token_type_ids":batch[2],
                "labels":batch[3]
            }

            outputs1 = model_f1(**inputs)
            outputs2 = model_f2(**inputs)

            logits1 = outputs1[1]
            logits2 = outputs2[1]

            logits1 = softmax(logits1, dim=1)
            logits2 = softmax(logits2, dim=1)
        if len(all_logits1) == 0:
            all_logits1 = logits1.detach().cpu().numpy()
            all_logits2 = logits2.detach().cpu().numpy()
            all_true_labels = batch[3].detach().cpu().numpy()

        else:
            all_logits1 = np.append(all_logits1, logits1.detach().cpu().numpy(), axis=0)
            all_logits2 = np.append(all_logits2, logits2.detach().cpu().numpy(),
            axis=0)
            all_true_labels = np.append(all_true_labels, batch[3].detach().cpu().numpy(), axis=0)
        
       # do collect
    all_preds_max_1 = np.max(all_logits1, axis=-1)
    # [batch_size, seq_len, 1]
    all_preds_max_2 = np.max(all_logits2, axis=-1)
# [batch_size, seq_len, 1]

    all_labels_1 = np.argmax(all_logits1, axis=1)
    all_labels_2 = np.argmax(all_logits2, axis=1)

    try:
        assert len(dataset) == len(all_preds_max_1) == len(all_preds_max_2) 
    except Exception as e:
        print("num of dataset:", len(dataset))
        print("num of all preds max 1", len(all_preds_max_1))
        print("num of all preds max 2", len(all_preds_max_2))

    labeled_data = []

    all_pseudo_true_labels = []
    for i in range(len(dataset)):
        record = cand_data[i]
        max_1 = all_preds_max_1[i]
        max_2 = all_preds_max_2[i]

        labels_1 = all_labels_1[i]
        labels_2 = all_labels_2[i]

        if labels_1 == labels_2 and max(max_1, max_2) >= args.alpha:
            record.label_ids = labels_1
            labeled_data.append(cand_data[i])
            all_pseudo_labels.append(labels_1)
            all_pseudo_true_labels.append(all_true_labels[i])
    
    if len(args.output_dir) > 0:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        # each line contains true & predict
        f1_predict_path = os.path.join(args.output_dir, "f1_results.txt")
        f2_predict_path = os.path.join(args.output_dir, "f2_results.txt")
        pseudo_predict_path = os.path.join(args.output_dir, "pseudo_labels.txt")
        num_labels_path = os.path.join(args.output_dir, "num_labels.txt")
        pseudo_acc_path = os.path.join(args.output_dir, "pseudo_acc.txt")
        with open(f1_predict_path, "w+", encoding="utf-8") as f:
            assert len(all_true_labels) == len(all_labels_1)
            for t, p in zip(all_true_labels, all_labels_1):
                line = str(t) + "\t" + str(p) + "\n"
                f.write(line)
        with open(f2_predict_path, "w+", encoding="utf-8") as f:
            assert len(all_true_labels) == len(all_labels_2)
            for t, p in zip(all_true_labels, all_labels_2):
                line = str(t) + "\t" + str(p) + "\n"
                f.write(line)
        with open(pseudo_predict_path, "w+", encoding="utf-8") as f:
            assert len(all_pseudo_true_labels) == len(all_pseudo_labels)
            for t, p in zip(all_pseudo_true_labels, all_pseudo_labels):
                line = str(t) + '\t' + str(p) + "\n"
                f.write(line)
        
        with open(pseudo_acc_path, "w+", encoding="utf-8") as f:
            assert len(all_pseudo_true_labels) == len(all_pseudo_labels)
            f.write( "{}\n".format(accuracy_score(all_pseudo_true_labels, all_pseudo_labels)))
        
        
    logger.info("**** collect labeled data size %s", len(labeled_data))
    logger.info("**** accuracy of pseudo labels: {}".format(accuracy_score(all_pseudo_true_labels, all_pseudo_labels)))
    return labeled_data

def prepare_dataset(source_features, labeled_features):
    features_L = source_features + labeled_features

    all_input_ids = torch.tensor([f.input_ids for f in features_L], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features_L], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features_L], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features_L], dtype=torch.long)

    dataset_L = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)


    all_input_ids = torch.tensor([f.input_ids for f in labeled_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in labeled_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in labeled_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in labeled_features], dtype=torch.long)


    dataset_TL =  TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    return dataset_L, dataset_TL


def tri_train(args, model_f1, model_f2, model_ft, source_features, target_features):
    Nt = args.N_init
    labeled_features = labelling(args, target_features, model_f1, model_f2, Nt)

    dataset_L, dataset_TL = prepare_dataset(source_features, labeled_features)
    k_step = args.k_step
    k_iterator = trange(k_step, desc="k_iter", disable=args.local_rank not in[-1, 0])

    cnt = 0
    
    model_ft = train_ft(args,model_ft, dataset_TL)

    result = test(args, model_ft, args.tokenizer, args.labels, args.pad_token_label_id, mode="test")
    result = evaluate(args, model_ft, args.tokenizer, args.labels, args.pad_token_label_id, mode="test")
    
    return model_f1, model_f2, model_ft

def joint_tri_train(args, model_f1, model_f2, model_ft, source_features, target_features):
    Nt = args.N_init

    all_input_ids = torch.tensor([f.input_ids for f in source_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in source_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in source_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in source_features], dtype=torch.long)


    dataset_S = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    model_f1, model_f2 = train_f1_f2(args, model_f1, model_f2, dataset_S)

    labeled_features = labelling(args, target_features, model_f1, model_f2, Nt)
    dataset_L, dataset_TL = prepare_dataset(source_features, labeled_features)
    model_ft = train_ft(args,model_ft, dataset_TL)
    result = test(args, model_ft, args.tokenizer, args.labels, args.pad_token_label_id, mode="test")
    result = evaluate(args, model_ft, args.tokenizer, args.labels, args.pad_token_label_id, mode="test")
    return model_f1, model_f2, model_ft
    

def train_f1_f2(args, model_f1, model_f2, train_dataset):
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()
    
    args.train_batch_size = args.mini_batch_size * max(1, args.n_gpu)
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
    logger.info("***** train f1 f2 ******")
    logger.info("***** Num examples: {} ********".format(len(train_dataset)))

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
            loss1 = outputs1[0]

            
            outputs2 = model_f2(**inputs)
            loss2 = outputs2[0]

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
    
    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    tr_loss, logging_loss = 0.0, 0.0

    model_ft.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )

    set_seed(args)
    logger.info("******* train ft *************")
    for _ in range(1):
        epoch_iterator = tqdm(train_dataloader, desc="Iter(loss=X.XXX, lr=X.XXXXXXXX)", disable=args.local_rank not in [-1, 0])

        for step, batch in enumerate(epoch_iterator):
            model_ft.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "input_ids":batch[0],
                "attention_mask":batch[1],
                "token_type_ids":batch[2],
                "labels":batch[3],
            }

            outputs = model_ft(**inputs)
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
                    torch.nn.utils.clip_grad_norm_(model_ft.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model_ft.zero_grad()
                global_step += 1
   
                
    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return model_ft

def train(args, train_dataset, model, tokenizer, labels, pad_token_label_id):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    if args.warmup_ratio > 0:
        args.warmup_steps = int(t_total * args.warmup_ratio)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
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
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        try:
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iter(loss=X.XXX, lr=X.XXXXXXXX)", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet"] else None
                )  # XLM and RoBERTa don"t use segment_ids

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
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
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev")
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step
    

def evaluate(args, model, tokenizer, labels, pad_token_label_id, mode, prefix=""):
    eval_examples = args.processor.get_dev_examples(args.data_dir, args.source_task)
    
    eval_features = convert_examples_to_features(
        eval_examples, labels, args.max_seq_length, tokenizer
    )
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in eval_features], dtype=torch.long)

    eval_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)

    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    label_map = {v:i for i,v in enumerate(labels)}

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps, nb_eval_examples = 0, 0
    preds = None
    eval_TP, eval_FP, eval_FN = 0, 0, 0
    eval_accuracy = 0
    model.eval()
    all_labels = []
    all_preds = []
    all_false_prob = []
    softmax = torch.nn.Softmax(dim=-1)
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask":batch[1],
                "token_type_ids":batch[2],
                "labels": batch[3],
                }

            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            if args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        
        
        label_ids = batch[3].detach().cpu().numpy()
        if len(all_labels) == 0:
            all_labels = label_ids
        else:
            all_labels = np.append(all_labels, label_ids)
        
        logits = torch.nn.functional.log_softmax(logits, dim=-1)
        logits = logits.detach().cpu().numpy()
        
        preds = np.argmax(logits, axis= -1)

        if len(all_preds) == 0:
            all_preds = preds
        else:
            all_preds = np.append(all_preds, preds)
           


    eval_loss = eval_loss / nb_eval_steps

    results = {
        "task": args.source_task,
        "loss": eval_loss,
        "eval_accuracy": accuracy_score(all_labels, all_preds),
        
    }

    logger.info("***** Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    return results

def test(args, model, tokenizer, labels, pad_token_label_id, mode, prefix=""):
    eval_examples = args.processor.get_dev_examples(args.data_dir, args.target_task)
    
    eval_features = convert_examples_to_features(
        eval_examples, labels, args.max_seq_length, tokenizer
    )
       
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in eval_features], dtype=torch.long)
  
    eval_dataset = TensorDataset(all_input_ids, all_input_mask,  all_segment_ids, all_label_ids)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    label_map = {v:i for i,v in enumerate(labels)}

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps, nb_eval_examples = 0, 0
    preds = None
    eval_TP, eval_FP, eval_FN = 0, 0, 0
    eval_accuracy = 0
    model.eval()
    softmax = torch.nn.Softmax(dim=-1)
    
    all_labels = []
    all_preds = []
    all_false_prob = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask":batch[1],
                "token_type_ids":batch[2],
                "labels": batch[3],
                }

            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            if args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        
        
        label_ids = batch[3].detach().cpu().numpy()
        if len(all_labels) == 0:
            all_labels = label_ids

        else:
            all_labels = np.append(all_labels, label_ids)

        logits = torch.nn.functional.log_softmax(logits, dim=-1)
        logits = logits.detach().cpu().numpy()
        
        preds = np.argmax(logits, axis= -1)

        if len(all_preds) == 0:
            all_preds = preds
        else:
            all_preds = np.append(all_preds, preds)

    eval_loss = eval_loss / nb_eval_steps

    results = {
        "task": args.target_task,
        "loss": eval_loss,
        "eval_accuracy": accuracy_score(all_labels, all_preds),
    }

    logger.info("***** Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    if len(args.result_path) > 0:
        txt_results = {}
        import json
        with open(args.result_path, "w+", encoding="utf-8") as f:
            txt_results["source_task"] = args.source_task
            txt_results["target_task"] = results["task"]
            txt_results["acc"] = results["eval_accuracy"]
            json.dump(txt_results, f)

    return results



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
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--model_f1_path",
        default=None,
        type=str,
        required=True,
        )
    parser.add_argument(
        "--model_f2_path",
        default=None,
        type=str,
        required=True
    )
    parser.add_argument(
        "--model_ft_path",
        default=None,
        type=str,
        required=True
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

    parser.add_argument("--result_path", type=str, default="")
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
    parser.add_argument(
        "--supervised_training",
        action="store_true",
    )
    parser.add_argument(
        "--finetuned_bert",
        action="store_true"
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")

    parser.add_argument("--source_task", type=str, default="")
    parser.add_argument("--target_task", type=str, default="")

    parser.add_argument("--k_step", type=int, default=1)
    parser.add_argument("--iter", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--N_init", type=int, default=100)
    parser.add_argument("--mini_batch_size", type=int, default=32)
    parser.add_argument("--result_dir", type=str, default="")
    parser.add_argument("--joint_loss", action="store_true")

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
    processor = DataProcessor()
    args.processor = processor
    labels = processor.get_labels(args.data_dir)
    num_labels = len(labels)

    pad_token_label_id = CrossEntropyLoss().ignore_index
    args.pad_token_label_id = pad_token_label_id
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

  
    args.tokenizer = tokenizer
    args.labels = labels
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    
    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:

        source_examples = processor.get_train_examples(args.data_dir, args.source_task)
        target_examples = processor.get_train_examples(args.data_dir, args.target_task)

        source_features = convert_examples_to_features(source_examples, labels, args.max_seq_length, tokenizer)
        target_features = convert_examples_to_features(target_examples, labels, args.max_seq_length, tokenizer)

        args.source_features = source_features
        args.target_features = target_features

        logger.info("**** source examples: {} *******".format(len(source_features)))
        logger.info("****  target examples: {} *****".format(len(target_features)))

        model_f1 = AutoModelForSequenceClassification.from_pretrained(
            args.model_f1_path,
            from_tf=bool(".ckpt" in args.model_f1_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

        model_f2 = AutoModelForSequenceClassification.from_pretrained(
            args.model_f2_path,
            from_tf=bool(".ckpt" in args.model_f1_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

        model_ft = AutoModelForSequenceClassification.from_pretrained(
            args.model_ft_path,
            from_tf=bool(".ckpt" in args.model_f1_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

        model_f1.to(args.device)
        model_f2.to(args.device)
        model_ft.to(args.device)
        if args.joint_loss:
            tri_train_func = joint_tri_train
        else:
            tri_train_func = tri_train
        model_f1, model_f2, model_ft = tri_train_func(args, model_f1, model_f2, model_ft, source_features, target_features)
        model = model_ft

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
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
            model.to(args.device)
            result= evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev", prefix=global_step)
            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
            results.update(result)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))

    if args.do_predict and args.local_rank in [-1, 0]:
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir, **tokenizer_args)
        model = AutoModelForSequenceClassification.from_pretrained(args.output_dir)
        model.to(args.device)
        result = test(args, model, tokenizer, labels, pad_token_label_id, mode="test")
        # Save results
        output_test_results_file = os.path.join(args.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))
    
    return results

if __name__ == "__main__":
    main()
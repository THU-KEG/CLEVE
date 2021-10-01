import argparse
import glob
import logging
import os
import random

import json
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    CONFIG_NAME,
    AdamW,
    BertConfig,
    BertForMultipleChoice,
    BertTokenizer,
    RobertaConfig,
    RobertaForMultipleChoice,
    RobertaTokenizer,
    RobertaModel,
    XLNetConfig,
    XLNetForMultipleChoice,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
)
from utils_ee import convert_examples_to_features, processors, convert_contrast_examples_to_features
from sklearn.metrics import f1_score,precision_score,recall_score
from model import DMBERT, DMRoBERTa, BERTContrastive, RoBERTaContrastive

#try:
#    from torch.utils.tensorboard import SummaryWriter
#except ImportError:
#    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, RobertaConfig)), ()
)

MODEL_CLASSES = {
    "bert": (BertConfig, DMBERT, BertTokenizer, BERTContrastive),
    # "xlnet": (XLNetConfig, XLNetForMultipleChoice, XLNetTokenizer),
    "roberta": (RobertaConfig, RobertaModel, RobertaTokenizer, RobertaModel),
}

def calculate_scores(preds, labels, dimE):
    positive_labels=list(range(1,dimE)) #assume 0 is NA
    pre = precision_score(labels, preds, labels=positive_labels, average='micro')
    recall = recall_score(labels, preds, labels=positive_labels, average='micro')
    f1 = f1_score(labels, preds, labels=positive_labels, average='micro')
    return pre, recall, f1

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer,):
    """ Train the model """
    #if args.local_rank in [-1, 0]:
        #tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

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
    tr_loss, logging_loss = 0.0, 0.0
    best_dev_f1 = 0.0
    best_steps = 0
    model.zero_grad()
    best_dev_preds = []
    best_test_preds = []
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2]
                if args.model_type in ["bert", "xlnet"]
                else None,  # XLM don't use segment_ids
                "maskL": batch[3],
                "maskR": batch[4],
                "labels": batch[5],
            }
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results, dev_pred_results = evaluate(args, model, tokenizer)
                        #for key, value in results.items():
                            #tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                        if results["eval_f1"] > best_dev_f1:
                            best_dev_f1 = results["eval_f1"]
                            best_steps = global_step
                            best_dev_preds = dev_pred_results
                            if args.do_test:
                                results_test,test_pred_results = evaluate(args, model, tokenizer, test=True)
                                best_test_preds = test_pred_results
                                #for key, value in results_test.items():
                                    #tb_writer.add_scalar("test_{}".format(key), value, global_step)
                                logger.info(
                                    "test f1: %s, loss: %s, global steps: %s",
                                    str(results_test["eval_f1"]),
                                    str(results_test["eval_loss"]),
                                    str(global_step),
                                )
                                with open(args.output_dir+'/best.txt','w') as f:
                                    f.write("best F1: {}\n".format(str(results_test["eval_f1"])))
                                    f.write("best p: {}\n".format(str(results_test["eval_p"])))
                                    f.write("best recall: {}\n".format(str(results_test["eval_recall"])))
                                    f.write("best loss: {}\n".format(str(results_test["eval_loss"])))
                                    f.write("best step: {}\n".format(str(best_steps)))
                                
                                with open(args.output_dir+'/pred.json','w') as f:
                                    json.dump({'dev':best_dev_preds,'test':best_test_preds},f)

                    #tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    #tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logger.info(
                        "Average loss: %s at global step: %s",
                        str((tr_loss - logging_loss) / args.logging_steps),
                        str(global_step),
                    )
                    logging_loss = tr_loss

                # if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                #     # Save model checkpoint
                #     output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                #     if not os.path.exists(output_dir):
                #         os.makedirs(output_dir)
                #     model_to_save = (
                #         model.module if hasattr(model, "module") else model
                #     )  # Take care of distributed/parallel training
                #     model_to_save.save_pretrained(output_dir)
                #     tokenizer.save_vocabulary(output_dir)
                #     torch.save(args, os.path.join(output_dir, "training_args.bin"))
                #     logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    #if args.local_rank in [-1, 0]:
        #tb_writer.close()

    return global_step, tr_loss / global_step, best_steps

def pretrain(args, train_dataset, model, tokenizer, test_dataset):
    """ Train the model """
    #if args.local_rank in [-1, 0]:
        #tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

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
    tr_loss, logging_loss = 0.0, 0.0
    con_tr_loss, con_logging_loss = 0.0, 0.0
    lm_tr_loss, lm_logging_loss = 0.0, 0.0
    best_dev_f1 = 0.0
    min_eval_loss = -1
    best_steps = 0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2]
                if args.model_type in ["bert", "xlnet"]
                else None,  # XLM don't use segment_ids
                "trigger_mask": batch[3],
                "arg_mask": batch[4],
                "none_arg_mask": batch[5],
                "none_arg_length_mask":batch[6],
                'masked_lm_labels': batch[7],
            }
            outputs = model(**inputs)
            # loss = outputs[0] + outputs[1]  # model outputs are always tuple in transformers (see doc)

            con_loss = outputs[0]
            lm_loss = outputs[1]
            loss = con_loss + lm_loss

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
                con_loss = con_loss.mean()
                lm_loss = lm_loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                con_loss = con_loss / args.gradient_accumulation_steps
                lm_loss = lm_loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            con_tr_loss += con_loss.item()
            lm_tr_loss += lm_loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    
                    eval_loss = pretrain_evaluate(args,test_dataset,model,tokenizer)

                    logger.info(
                        "Average loss: %s, con_loss: %s, lm_loss: %s, eval_loss: %s, at global step: %s",
                        str((tr_loss - logging_loss) / args.logging_steps),
                        str((con_tr_loss - con_logging_loss) / args.logging_steps),
                        str((lm_tr_loss - lm_logging_loss) / args.logging_steps),
                        str(eval_loss),
                        str(global_step),
                    )
                    logging_loss = tr_loss
                    con_logging_loss = con_tr_loss
                    lm_logging_loss = lm_tr_loss
                    if eval_loss < min_eval_loss or min_eval_loss==-1:
                        best_steps = global_step+0
                        min_eval_loss = eval_loss
                        if not os.path.exists(args.output_dir):
                            os.makedirs(args.output_dir)
                        with open(os.path.join(args.output_dir, "best_steps.txt"),'w') as f:
                            f.write('best step at:' + str(best_steps)+'\n')
                            f.write('eval_loss:' + str(eval_loss))

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module.roberta if hasattr(model, "module") else model.roberta
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        
       
        
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    #if args.local_rank in [-1, 0]:
        #tb_writer.close()

    return global_step, tr_loss / global_step, best_steps, min_eval_loss

def pretrain_evaluate(args, eval_dataset, model, tokenizer, prefix="", test=False):

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    eval_loss = 0.0
    step = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2]
                if args.model_type in ["bert", "xlnet"]
                else None,  # XLM don't use segment_ids
                "trigger_mask": batch[3],
                "arg_mask": batch[4],
                "none_arg_mask": batch[5],
                "none_arg_length_mask":batch[6],
                'masked_lm_labels': batch[7],
            }
            outputs = model(**inputs)
            eval_loss += (outputs[0] + outputs[1]).mean().item()
        step+=1

    return eval_loss/step


def evaluate(args, model, tokenizer, prefix="", test=False):
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    pred_results = []
    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset, eval_ids = load_and_cache_examples(args, eval_task, tokenizer, evaluate=not test, test=test)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu evaluate
        #if args.n_gpu > 1:
            #print("?????",args.n_gpu)
            #model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2]
                    if args.model_type in ["bert", "xlnet"]
                    else None,  # XLM don't use segment_ids
                    "maskL": batch[3],
                    "maskR": batch[4],
                    "labels": batch[5],
                }
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=1)
        #print(eval_task)
        #print(processors[eval_task])
        pred_results.extend(preds.tolist())
        precision, recall, f1 = calculate_scores(preds, out_label_ids, len(processors[eval_task]().get_labels()))
        result = {"eval_p": precision, "eval_recall": recall, "eval_f1": f1, "eval_loss": eval_loss}
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, "is_test_" + str(test).lower() + "_eval_results.txt")

        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(str(prefix) + " is test:" + str(test)))
            writer.write("model           =%s\n" % str(args.model_name_or_path))
            writer.write(
                "total batch size=%d\n"
                % (
                    args.per_gpu_train_batch_size
                    * args.gradient_accumulation_steps
                    * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)
                )
            )
            writer.write("train num epochs=%d\n" % args.num_train_epochs)
            writer.write("fp16            =%s\n" % args.fp16)
            writer.write("max seq length  =%d\n" % args.max_seq_length)
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
        assert len(eval_ids)==len(pred_results)
    return results, dict(list(zip(eval_ids,pred_results)))

def load_and_cache_contrast_examples(args, task, tokenizer):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    # Load data features from cache or dataset file
    cached_mode = "pretrain_contrast"
    cached_features_file = os.path.join(
        args.data_dir,
        "contrast_cached_{}_{}_{}_{}".format(
            cached_mode,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading contrast features from cached file %s", cached_features_file)
        train_contrast_features = torch.load(cached_features_file)
        test_contrast_features = torch.load(cached_features_file+'_test')
    else:
        logger.info("Creating contrast features from dataset file at %s", args.data_dir)

        contrast_examples = processor.get_contrast_examples(args.data_dir)
        
        contrast_features = convert_contrast_examples_to_features(
            contrast_examples,
            args.max_seq_length,
            args.max_contrast_entity_per_sentence,
            tokenizer,
            pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        )

        feature_len = int(len(contrast_features) / 10 * 8)

        train_contrast_features = contrast_features[:feature_len]
        test_contrast_features = contrast_features[feature_len:]

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(train_contrast_features, cached_features_file)
            torch.save(test_contrast_features, cached_features_file+"_test")

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in train_contrast_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_contrast_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_contrast_features], dtype=torch.long)
    all_trigger_mask = torch.tensor([f.trigger_mask for f in train_contrast_features], dtype=torch.float)
    all_arg_mask = torch.tensor([f.arg_mask for f in train_contrast_features], dtype=torch.float)
    all_none_arg_mask = torch.tensor([f.none_arg_mask for f in train_contrast_features], dtype=torch.float)
    all_none_arg_length_mask = torch.tensor([f.none_arg_length_mask for f in train_contrast_features], dtype=torch.float)
    all_lm_masked_labels = torch.tensor([f.lm_masked_labels for f in train_contrast_features], dtype=torch.long)

    train_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_trigger_mask, all_arg_mask, all_none_arg_mask, all_none_arg_length_mask,all_lm_masked_labels)
    
    all_input_ids = torch.tensor([f.input_ids for f in test_contrast_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_contrast_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in test_contrast_features], dtype=torch.long)
    all_trigger_mask = torch.tensor([f.trigger_mask for f in test_contrast_features], dtype=torch.float)
    all_arg_mask = torch.tensor([f.arg_mask for f in test_contrast_features], dtype=torch.float)
    all_none_arg_mask = torch.tensor([f.none_arg_mask for f in test_contrast_features], dtype=torch.float)
    all_none_arg_length_mask = torch.tensor([f.none_arg_length_mask for f in test_contrast_features], dtype=torch.float)
    all_lm_masked_labels = torch.tensor([f.lm_masked_labels for f in test_contrast_features], dtype=torch.long)

    test_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_trigger_mask, all_arg_mask, all_none_arg_mask, all_none_arg_length_mask,all_lm_masked_labels)
    
    return train_dataset,test_dataset

def load_and_cache_examples(args, task, tokenizer, evaluate=False, test=False):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    # Load data features from cache or dataset file
    if evaluate:
        cached_mode = "dev"
    elif test:
        cached_mode = "test"
    else:
        cached_mode = "train"
    assert not (evaluate and test)
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            cached_mode,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if evaluate:
            examples = processor.get_dev_examples(args.data_dir)
        elif test:
            examples = processor.get_test_examples(args.data_dir)
        else:
            examples = processor.get_train_examples(args.data_dir)
        logger.info("Training number: %s", str(len(examples)))

        features = convert_examples_to_features(
            examples,
            label_list,
            args.max_seq_length,
            tokenizer,
            pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_maskL = torch.tensor([f.maskL for f in features], dtype=torch.float)
    all_maskR = torch.tensor([f.maskR for f in features], dtype=torch.float)
    all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_maskL, all_maskR, all_label_ids)
    if evaluate or test:
        return dataset, [f.example_id for f in features]
    else:
        return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    
    parser.add_argument(
        "--max_contrast_entity_per_sentence",
        default=10,
        type=int,
        required=True,
        help="Max contrast entity per sentence in contrastive pretraining stage.",
    )


    # Other parameters
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
    parser.add_argument("--do_pretrain", action="store_true", help="Whether to pretrain using MI.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run test on the test set")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    
    parser.add_argument("--per_gpu_positive_batch_size",default=1,type=int,help="Batch size per GPU/CPU for positive pretrain")
    parser.add_argument("--per_gpu_negativex_batch_size",default=3,type=int,help="Batch size per GPU/CPU for negativex pretrain")
    parser.add_argument("--per_gpu_negativey_batch_size",default=3,type=int,help="Batch size per GPU/CPU for negativey pretrain")
    parser.add_argument(
        "--num_pretrain_epochs", default=3.0, type=float, help="Total number of pretraining epochs to perform."
    )
    parser.add_argument(
        "--positive_max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of pretraining steps to perform. Override num_pretrain_epochs.",
    )

    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
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
    

    """
        Input Error Check
    """

    assert os.path.exists(args.data_dir), ValueError('data_dir does not exist!')
    assert args.task_name=='ace', ValueError('task_name is not supported, please use ace')
    check_suffix = 'json' if args.task_name=='ace' else 'jsonl'
    assert 'train.{}'.format(check_suffix) in os.listdir(args.data_dir), ValueError('train file does not exist!')
    assert 'dev.{}'.format(check_suffix) in os.listdir(args.data_dir), ValueError('dev file does not exist!')
    assert 'test.{}'.format(check_suffix) in os.listdir(args.data_dir), ValueError('test file does not exist!')

    if not args.do_pretrain:
        assert os.path.exists(args.model_name_or_path), ValueError('Model path does not exists!')
        check_model_files = os.listdir(args.model_name_or_path)
        assert 'config.json' in check_model_files, ValueError('Model files are not complete! (config.json is missing)')
        assert 'merges.txt' in check_model_files, ValueError('Model files are not complete! (merges.txt is missing)')
        assert 'pytorch_model.bin' in check_model_files, ValueError('Model files are not complete! (pytorch_model.bin is missing)')
        assert 'special_tokens_map.json' in check_model_files, ValueError('Model files are not complete! (special_tokens_map.json is missing)')
        assert 'tokenizer_config.json' in check_model_files, ValueError('Model files are not complete! (tokenizer_config.json is missing)')
        assert 'training_args.bin' in check_model_files, ValueError('Model files are not complete! (training_args.bin is missing)')
        assert 'vocab.json' in check_model_files, ValueError('Model files are not complete! (vocab.json is missing)')

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
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

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class, contrastive_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.do_pretrain:
        if args.model_type=='roberta':
            model = contrastive_class.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),#True,
                config=config,
                cache_dir=args.cache_dir if args.cache_dir else None
                )
            model = RoBERTaContrastive(config,model)
        else:
            raise ValueError('model_type should be roberta')
    else:
        if args.model_type=='roberta':
            model = model_class.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),#True,
                config=config,
                cache_dir=args.cache_dir if args.cache_dir else None
            )
            model = DMRoBERTa(config,model)
        else:
            raise ValueError('model_type should be roberta')

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    best_steps = 0

    if args.do_pretrain:
        print("--do pretrain--")
        pretrain_dataset, pretrain_dataset_test = load_and_cache_contrast_examples(args, args.task_name, tokenizer)
        global_step, tr_loss ,best_steps, min_loss = pretrain(args, pretrain_dataset, model, tokenizer,pretrain_dataset_test)
        logger.info(" contrast_global_step = %s, contrast_average loss = %s, best_steps = %s, min_loss = %s", global_step, tr_loss, best_steps, min_loss)
        return

    # Trainingr
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss, best_steps = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s, best_steps = %s", global_step, tr_loss, best_steps)

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

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        if not args.do_train:
            args.output_dir = args.model_name_or_path
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result,_ = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    if args.do_test and args.local_rank in [-1, 0]:
        if not args.do_train:
            args.output_dir = args.model_name_or_path
        checkpoints = [args.output_dir]
        # if args.eval_all_checkpoints: # can not use this to do test!!
        #     checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
        #     logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result,_ = evaluate(args, model, tokenizer, prefix=prefix, test=True)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)
    if best_steps:
        logger.info("best steps of eval f1 is the following checkpoints: %s", best_steps)
    return results


if __name__ == "__main__":
    main()

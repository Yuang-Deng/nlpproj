# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb

import argparse
import csv
import json
import logging
import os
import pickle
import random

import numpy as np
import torch
from google_albert_pytorch_modeling import AlbertConfig, AlbertForMultipleChoice
from pytorch_modeling import BertConfig, BertForMultipleChoice, ALBertConfig, ALBertForMultipleChoice, BertForMultipleChoiceWithMatch
from tools import official_tokenization as tokenization
from tools import utils
from tools.pytorch_optimization import get_optimization
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

n_class = 4
reverse_order = False
sa_step = False

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class C3Example(object):
    """A single training/test example for the C3 dataset."""

    def __init__(self,
                 C3_id,
                 context_sentence,
                 start_ending,
                 ending_0,
                 ending_1,
                 ending_2,
                 ending_3,
                 label=None):
        self.C3_id = C3_id
        self.context_sentence = context_sentence
        self.start_ending = start_ending
        self.endings = [
            ending_0,
            ending_1,
            ending_2,
            ending_3,
        ]
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            f"C3_id: {self.C3_id}",
            f"context_sentence: {self.context_sentence}",
            f"start_ending: {self.start_ending}",
            f"ending_0: {self.endings[0]}",
            f"ending_1: {self.endings[1]}",
            f"ending_2: {self.endings[2]}",
            f"ending_3: {self.endings[3]}",
        ]

        if self.label is not None:
            l.append(f"label: {self.label}")

        return ", ".join(l)

class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label

                 ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
                'doc_len': doc_len,
                'ques_len': ques_len,
                'option_len': option_len
            }
            for _, input_ids, input_mask, segment_ids, doc_len, ques_len, option_len in choices_features
        ]
        self.label = label

def read_race(path):
    article = []
    question = []
    ct1 = []
    ct2 = []
    ct3 = []
    ct4 = []
    y = []
    q_id = []
    for p in path:
        with open(p, 'r', encoding='utf_8') as f:
            data_all = json.load(f)
            for instance in data_all:
                for j in range(len(instance[1])):
                    ans = 0
                    for k in range(len(instance[1][j]["choice"])):
                        if "answer" in instance[1][j].keys() and instance[1][0]["choice"][k] == instance[1][j]['answer']:
                            ans = k
                        eval("ct"+str(k+1)).append(instance[1][j]["choice"][k])
                    for k in range(len(instance[1][j]["choice"]), 4):
                        eval("ct"+str(k+1)).append('无效答案')  # 有些C3数据选项不足4个，添加[无效答案]能够有效增强模型收敛稳定性
                    question.append(instance[1][j]['question'])
                    if "id" in instance[1][j].keys():
                        q_id.append(instance[1][j]['id'])
                    else:
                        q_id.append(0)
                    art = instance[0]
                    l = []
                    for i in art:
                        l.append(i)
                    article.append(' '.join(l))
                    y.append(ans)
    return article, question, ct1, ct2, ct3, ct4, y, q_id


def read_C3_examples(input_file, is_training):
    if is_training:
        article, question, ct1, ct2, ct3, ct4, y, q_id = read_race(input_file)

    examples = [
        C3Example(
            C3_id=s8,
            context_sentence=s1,
            start_ending=s2,  # in the C3 dataset, the
            # common beginning of each
            # choice is stored in "sent2".
            ending_0=s3,
            ending_1=s4,
            ending_2=s5,
            ending_3=s6,
            label=s7 if is_training else None
        ) for i, (s1, s2, s3, s4, s5, s6, s7, s8), in enumerate(zip(article, question, ct1, ct2, ct3, ct4, y, q_id))
    ]

    return examples

def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 is_training):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    ool = 0
    for example_index, example in enumerate(examples):
        context_tokens = tokenizer.tokenize(example.context_sentence)
        start_ending_tokens = tokenizer.tokenize(example.start_ending)

        choices_features = []
        for ending_index, ending in enumerate(example.endings):
            # We create a copy of the context tokens in order to be
            # able to shrink it according to ending_tokens
            context_tokens_choice = context_tokens[:]  # + start_ending_tokens

            ending_token = tokenizer.tokenize(ending)
            option_len = len(ending_token)
            ques_len = len(start_ending_tokens)

            ending_tokens = start_ending_tokens + ending_token

            # Modifies `context_tokens_choice` and `ending_tokens` in
            # place so that the total length is less than the
            # specified length.  Account for [CLS], [SEP], [SEP] with
            # "- 3"
            # ending_tokens = start_ending_tokens + ending_tokens
            _truncate_seq_pair(context_tokens_choice, ending_tokens, max_seq_length - 3)
            doc_len = len(context_tokens_choice)
            if len(ending_tokens) + len(context_tokens_choice) >= max_seq_length - 3:
                ques_len = len(ending_tokens) - option_len

            tokens = ["[CLS]"] + context_tokens_choice + ["[SEP]"] + ending_tokens + ["[SEP]"]
            segment_ids = [0] * (len(context_tokens_choice) + 2) + [1] * (len(ending_tokens) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            # assert (doc_len + ques_len + option_len) <= max_seq_length
            if (doc_len + ques_len + option_len) > max_seq_length:
                print(doc_len, ques_len, option_len, len(context_tokens_choice), len(ending_tokens))
                assert (doc_len + ques_len + option_len) <= max_seq_length
            choices_features.append((tokens, input_ids, input_mask, segment_ids, doc_len, ques_len, option_len))

        label = example.label
        if example_index < 5:
            logger.info("*** Example ***")
            logger.info(f"C3_id: {example.C3_id}")
            for choice_idx, (tokens, input_ids, input_mask, segment_ids, doc_len, ques_len, option_len) in enumerate(choices_features):
                logger.info(f"choice: {choice_idx}")
                logger.info(f"tokens: {' '.join(tokens)}")
                logger.info(f"input_ids: {' '.join(map(str, input_ids))}")
                logger.info(f"input_mask: {' '.join(map(str, input_mask))}")
                logger.info(f"segment_ids: {' '.join(map(str, segment_ids))}")
            if is_training:
                logger.info(f"label: {label}")

        features.append(
            InputFeatures(
                example_id=example.C3_id,
                choices_features=choices_features,
                label=label
            )
        )

    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence tuple in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) >= len(tokens_b) and len(tokens_a) >= len(tokens_c):
            tokens_a.pop()
        elif len(tokens_b) >= len(tokens_a) and len(tokens_b) >= len(tokens_c):
            tokens_b.pop()
        else:
            tokens_c.pop()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--gpu_ids",
                        default='0',
                        type=str,)
    # parser.add_argument("--data_dir",
    #                     default='C:\Users\Alex\WorkSpace\pycharm\CLUE\\baselines\models_pytorch\mrc_pytorch\mrc_data\c3',
    #                     type=str,
    #                     help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    # parser.add_argument("--task_name",
    #                     default='c3',
    #                     type=str)
    # parser.add_argument("--bert_config_file",
    #                     default='./check_points/pretrain_models/albert_large_zh/bert_config.json',
    #                     type=str,
    #                     help="The config json file corresponding to the pre-trained BERT model. \n"
    #                          "This specifies the model architecture.")
    # parser.add_argument("--vocab_file",
    #                     default='./check_points/pretrain_models/albert_large_zh/vocab.txt',
    #                     type=str,
    #                     help="The vocabulary file that the BERT model was trained on.")
    # parser.add_argument("--output_dir",
    #                     default='./c3/roberta_wwm_ext_base',
    #                     type=str,
    #                     help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    # parser.add_argument("--init_checkpoint",
    #                     default='check_points/pretrain_models/albert_xxlarge_google_zh_v1121/pytorch_model.pth',
    #                     type=str,
    #                     help="Initial checkpoint (usually from a pre-trained BERT model).")
    parser.add_argument("--do_lower_case",
                        default=True,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=True,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=True,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--schedule",
                        default='warmup_linear',
                        type=str,
                        help='schedule')
    parser.add_argument("--weight_decay_rate",
                        default=0.01,
                        type=float,
                        help='weight_decay_rate')
    parser.add_argument('--clip_norm',
                        type=float,
                        default=1.0)
    parser.add_argument("--num_train_epochs",
                        default=8.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--float16',
                        action='store_true',
                        default=False)
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=422,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=16,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--setting_file', type=str, default='setting.txt')
    parser.add_argument('--log_file', type=str, default='log.txt')

    args = parser.parse_args()
    args.data_dir = "./data/c3"
    args.train_m_file = "m-train.json"
    args.dev_m_file = "m-dev.json"
    args.train_d_file = "d-train.json"
    args.dev_d_file = "d-dev.json"
    args.test_file = "test.json"
    args.output_dir = "./check_points/c3/bert_wwm_ext_base"
    args.init_checkpoint = "./pretrained_model/roberta_large/pytorch_model.bin"
    args.vocab_file = "./pretrained_model/roberta_large/vocab.txt"
    args.bert_config_file = "./pretrained_model/roberta_large/bert_config.json"
    weight = 0.1

    # args.init_checkpoint = "C:\Users\Alex\WorkSpace\pycharm\CLUE\\baselines\models_pytorch\mrc_pytorch\check_points\pretrain_models\\albert_large_zh\pytorch_model.bin"
    # args.vocab_file = "C:\Users\Alex\WorkSpace\pycharm\CLUE\\baselines\models_pytorch\mrc_pytorch\check_points\pretrain_models\\albert_large_zh\\vocab.txt"
    # args.bert_config_file = "C:\Users\Alex\WorkSpace\pycharm\CLUE\baselines\models_pytorch\mrc_pytorch\check_points\pretrain_models\albert_large_zh\bert_config.json"
    args.setting_file = os.path.join(args.output_dir, args.setting_file)
    args.log_file = os.path.join(args.output_dir, args.log_file)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.setting_file, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        print('------------ Options -------------')
        for k in args.__dict__:
            v = args.__dict__[k]
            opt_file.write('%s: %s\n' % (str(k), str(v)))
            print('%s: %s' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')
        print('------------ End -------------')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    if os.path.exists(args.log_file):
        os.remove(args.log_file)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    tokenizer = tokenization.BertTokenizer(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)
    # tokenizer = tokenization.BertTokenizer.from_pretrained("bert-base-chinese", do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_examples = read_C3_examples([os.path.join(args.data_dir, args.train_m_file), os.path.join(args.data_dir, args.train_d_file)], is_training=True)
        num_train_steps = int(len(train_examples) / args.train_batch_size /
                              args.gradient_accumulation_steps * args.num_train_epochs)

    if 'albert' in args.bert_config_file:
        if 'google' in args.bert_config_file:
            bert_config = AlbertConfig.from_json_file(args.bert_config_file)
            model = AlbertForMultipleChoice(bert_config, num_choices=n_class)
        else:
            bert_config = ALBertConfig.from_json_file(args.bert_config_file)
            model = ALBertForMultipleChoice(bert_config, num_choices=n_class)
            # model = hub.Module(name='RoBERTa-wwm-ext-large')
    else:
        bert_config = BertConfig.from_json_file(args.bert_config_file)
        # model = BertForMultipleChoice(bert_config, num_choices=n_class)
        model = BertForMultipleChoiceWithMatch(bert_config, num_choices=n_class)
        # model = BertForMultipleChoiceWithMatch.from_pretrained("bert-base-chinese", num_choices=4)

    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
                args.max_seq_length, bert_config.max_position_embeddings))

    if args.init_checkpoint is not None:
        utils.torch_show_all_params(model)
        utils.torch_init_model(model, args.init_checkpoint)
    if args.float16:
        model.half()
    model.to(device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    optimizer = get_optimization(model=model,
                                 float16=args.float16,
                                 learning_rate=args.learning_rate,
                                 total_steps=num_train_steps,
                                 schedule=args.schedule,
                                 warmup_rate=args.warmup_proportion,
                                 max_grad_norm=args.clip_norm,
                                 weight_decay_rate=args.weight_decay_rate,
                                 opt_pooler=True)  # multi_choice must update pooler

    global_step = 0
    eval_dataloader = None
    if args.do_eval:
        eval_examples = read_C3_examples([os.path.join(args.data_dir, args.dev_m_file), os.path.join(args.data_dir, args.dev_d_file)], is_training=True)
        feature_dir = os.path.join(args.data_dir, 'dev_features{}.pkl'.format(args.max_seq_length))
        if os.path.exists(feature_dir):
            eval_features = pickle.load(open(feature_dir, 'rb'))
        else:
            eval_features = convert_examples_to_features(eval_examples, tokenizer, args.max_seq_length, True)
            with open(feature_dir, 'wb') as w:
                pickle.dump(eval_features, w)

        all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
        all_doc_len = torch.tensor(select_field(eval_features, 'doc_len'), dtype=torch.long)
        all_ques_len = torch.tensor(select_field(eval_features, 'ques_len'), dtype=torch.long)
        all_option_len = torch.tensor(select_field(eval_features, 'option_len'), dtype=torch.long)

        all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label, all_doc_len, all_ques_len,
                                   all_option_len)
        if args.local_rank == -1:
            eval_sampler = SequentialSampler(eval_data)
        else:
            eval_sampler = DistributedSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    if args.do_train:
        best_accuracy = 0

        feature_dir = os.path.join(args.data_dir, 'train_features{}.pkl'.format(args.max_seq_length))
        if os.path.exists(feature_dir):
            train_features = pickle.load(open(feature_dir, 'rb'))
        else:
            train_features = convert_examples_to_features(train_examples, tokenizer, args.max_seq_length, True)
            with open(feature_dir, 'wb') as w:
                pickle.dump(train_features, w)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        all_input_ids = torch.tensor(select_field(train_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(train_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(train_features, 'segment_ids'), dtype=torch.long)
        all_doc_len = torch.tensor(select_field(train_features, 'doc_len'), dtype=torch.long)
        all_ques_len = torch.tensor(select_field(train_features, 'ques_len'), dtype=torch.long)
        all_option_len = torch.tensor(select_field(train_features, 'option_len'), dtype=torch.long)

        all_label = torch.tensor([f.label for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label, all_doc_len, all_ques_len,
                                   all_option_len)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,
                                      drop_last=True)
        steps_per_epoch = int(num_train_steps / args.num_train_epochs)

        for ie in range(int(args.num_train_epochs)):
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            with tqdm(total=int(steps_per_epoch), desc='Epoch %d' % (ie + 1)) as pbar:
                for step, batch in enumerate(train_dataloader):
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, label_ids, doc_len, ques_len, option_len = batch
                    loss, match_loss = model(input_ids, segment_ids, input_mask, doc_len, ques_len, option_len, label_ids)
                    loss = loss + weight * match_loss
                    if n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu.
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                    tr_loss += loss.item()

                    if args.float16:
                        optimizer.backward(loss)
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used and handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_steps,
                                                                          args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    else:
                        loss.backward()

                    nb_tr_examples += input_ids.size(0)
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        optimizer.step()  # We have accumulated enought gradients
                        model.zero_grad()
                        global_step += 1
                        nb_tr_steps += 1
                        pbar.set_postfix({'loss': '{0:1.5f}'.format(tr_loss / (nb_tr_steps + 1e-5))})
                        pbar.update(1)

            if args.do_eval:
                model.eval()
                eval_loss, eval_accuracy = 0, 0
                nb_eval_steps, nb_eval_examples = 0, 0
                logits_all = []
                for input_ids, input_mask, segment_ids, label_ids, all_doc_len, all_ques_len, all_option_len in tqdm(
                        eval_dataloader, desc="Evaluating"):
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    all_doc_len = all_doc_len.to(device)
                    all_ques_len = all_ques_len.to(device)
                    all_option_len = all_option_len.to(device)
                    label_ids = label_ids.to(device)

                    with torch.no_grad():
                        tmp_eval_loss, match_loss = model(input_ids, segment_ids, input_mask, all_doc_len, all_ques_len,
                                              all_option_len, label_ids)
                        tmp_eval_loss = tmp_eval_loss + weight * match_loss
                        logits = model(input_ids, segment_ids, input_mask, all_doc_len, all_ques_len, all_option_len)
                    logits = logits.detach().cpu().numpy()
                    label_ids = label_ids.cpu().numpy()
                    for i in range(len(logits)):
                        logits_all += [logits[i]]

                    tmp_eval_accuracy = accuracy(logits, label_ids.reshape(-1))

                    eval_loss += tmp_eval_loss.mean().item()
                    eval_accuracy += tmp_eval_accuracy

                    nb_eval_examples += input_ids.size(0)
                    nb_eval_steps += 1

                eval_loss = eval_loss / nb_eval_steps
                eval_accuracy = eval_accuracy / nb_eval_examples

                if args.do_train:
                    result = {'eval_loss': eval_loss,
                              'eval_accuracy': eval_accuracy,
                              'global_step': global_step,
                              'loss': tr_loss / nb_tr_steps}
                else:
                    result = {'eval_loss': eval_loss,
                              'eval_accuracy': eval_accuracy}

                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))

                with open(args.log_file, 'a') as aw:
                    aw.write("-------------------global steps:{}-------------------\n".format(global_step))
                    aw.write(str(json.dumps(result, indent=2)) + '\n')

                if eval_accuracy >= best_accuracy:
                    torch.save(model.state_dict(), os.path.join(args.output_dir, "model_best.pt"))
                    best_accuracy = eval_accuracy

        model.load_state_dict(torch.load(os.path.join(args.output_dir, "model_best.pt")))
        torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))

    model.load_state_dict(torch.load(os.path.join(args.output_dir, "model.pt")))

    if args.do_eval:
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        logits_all = []
        for input_ids, input_mask, segment_ids, label_ids, all_doc_len, all_ques_len, all_option_len in tqdm(
                eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            all_doc_len = all_doc_len.to(device)
            all_ques_len = all_ques_len.to(device)
            all_option_len = all_option_len.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                tmp_eval_loss, match_loss = model(input_ids, segment_ids, input_mask, all_doc_len, all_ques_len, all_option_len,
                                      label_ids)
                tmp_eval_loss = tmp_eval_loss + weight * match_loss
                logits = model(input_ids, segment_ids, input_mask, all_doc_len, all_ques_len, all_option_len)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.cpu().numpy()
            for i in range(len(logits)):
                logits_all += [logits[i]]

            tmp_eval_accuracy = accuracy(logits, label_ids.reshape(-1))

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples

        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy}

        output_eval_file = os.path.join(args.output_dir, "results_dev.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
        output_eval_file = os.path.join(args.output_dir, "logits_dev.txt")
        with open(output_eval_file, "w") as f:
            for i in range(len(logits_all)):
                for j in range(len(logits_all[i])):
                    f.write(str(logits_all[i][j]))
                    if j == len(logits_all[i]) - 1:
                        f.write("\n")
                    else:
                        f.write(" ")

        test_examples = read_C3_examples([os.path.join(args.data_dir, args.test_file)], is_training=True)

        feature_dir = os.path.join(args.data_dir, 'test_features{}.pkl'.format(args.max_seq_length))
        if os.path.exists(feature_dir):
            test_features = pickle.load(open(feature_dir, 'rb'))
        else:
            test_features = convert_examples_to_features(test_examples, tokenizer, args.max_seq_length, False)
            with open(feature_dir, 'wb') as w:
                pickle.dump(test_features, w)

        logger.info("***** Running testing *****")
        logger.info("  Num examples = %d", len(test_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        all_id = []
        for i in test_features:
            all_id.append(i.example_id)

        all_input_ids = torch.tensor(select_field(test_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(test_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(test_features, 'segment_ids'), dtype=torch.long)
        all_doc_len = torch.tensor(select_field(test_features, 'doc_len'), dtype=torch.long)
        all_ques_len = torch.tensor(select_field(test_features, 'ques_len'), dtype=torch.long)
        all_option_len = torch.tensor(select_field(test_features, 'option_len'), dtype=torch.long)

        all_label = torch.tensor([f.label for f in test_features], dtype=torch.long)
        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label, all_doc_len, all_ques_len,
                                   all_option_len)

        if args.local_rank == -1:
            test_sampler = SequentialSampler(test_data)
        else:
            test_sampler = DistributedSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

        model.eval()
        test_loss, test_accuracy = 0, 0
        nb_test_steps, nb_test_examples = 0, 0
        logits_all = []
        label_all = []
        for input_ids, input_mask, segment_ids, label_ids, all_doc_len, all_ques_len, all_option_len in tqdm(
                test_dataloader, desc="Testing"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            all_doc_len = all_doc_len.to(device)
            all_ques_len = all_ques_len.to(device)
            all_option_len = all_option_len.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                tmp_test_loss, match_loss = model(input_ids, segment_ids, input_mask, all_doc_len, all_ques_len, all_option_len,
                                      label_ids)
                tmp_test_loss = tmp_test_loss + weight * match_loss
                logits = model(input_ids, segment_ids, input_mask, all_doc_len, all_ques_len, all_option_len)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            for i in range(len(logits)):
                logits_all += [logits[i]]
                label_all += [label_ids]

            tmp_test_accuracy = accuracy(logits, label_ids.reshape(-1))

            test_loss += tmp_test_loss.mean().item()
            test_accuracy += tmp_test_accuracy

            nb_test_examples += input_ids.size(0)
            nb_test_steps += 1

        test_loss = test_loss / nb_test_steps
        test_accuracy = test_accuracy / nb_test_examples

        result = {'test_loss': test_loss,
                  'test_accuracy': test_accuracy}

        output_test_file = os.path.join(args.output_dir, "results_test.txt")
        with open(output_test_file, "w") as writer:
            logger.info("***** Test results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
        output_test_file = os.path.join(args.output_dir, "logits_test.txt")
        with open(output_test_file, "w") as f:
            for i in range(len(logits_all)):
                for j in range(len(logits_all[i])):
                    f.write(str(logits_all[i][j]))
                    if j == len(logits_all[i]) - 1:
                        f.write("\n")
                    else:
                        f.write(" ")

        # the test submission order can't be changed
        submission_test = os.path.join(args.output_dir, "c3_predict.json")
        test_preds = [int(np.argmax(logits_)) for logits_ in logits_all]
        res = []
        for id, pred in zip(all_id, test_preds):
            res.append({"id":id, "label" : pred})
        res = sorted(res, key=lambda x : x["id"])
        with open(submission_test, "w") as f:
            # json.dump(test_preds, f)
            for r in res:
                f.writelines(json.dumps(r))
                f.write('\n')


if __name__ == "__main__":
    main()

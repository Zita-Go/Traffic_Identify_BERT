"""
This script provides an exmaple to wrap UER-py for classification.
"""
import sys
sys.path.append('D:/Users/ZitaGo/PycharmProjects/Transaction_analysis/traffic_identification/ET-BERT-main')

import random
import argparse
import torch
import torch.nn as nn
from uer.layers import *
from uer.encoders import *
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils import *
from uer.utils.optimizers import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model, merge_model
from uer.opts import finetune_opts
import tqdm
import json
import numpy as np
from lora_init import *

class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab), "encoder1")
        self.encoder1 = str2encoder[args.encoder1](args, "encoder1")
        self.encoder2 = str2encoder[args.encoder2](args, "encoder2")
        self.attr_dim = args.attr_dim
        if self.attr_dim > 0:
            self.attr_embedding = nn.Linear(self.attr_dim + args.hidden_size["encoder1"], args.emb_size["encoder2"])
        # self.encoder = str2encoder[args.encoder](args)
        self.labels_num = args.labels_num
        self.pooling1 = args.pooling1
        self.pooling2 = args.pooling2
        self.soft_targets = args.soft_targets
        self.soft_alpha = args.soft_alpha
        self.output_layer_1 = nn.Linear(args.hidden_size["encoder2"], args.hidden_size["encoder2"])
        self.output_layer_2 = nn.Linear(args.hidden_size["encoder2"], self.labels_num)

    def forward(self, src, tgt, seg, seg_flow, seq_attr=None, soft_tgt=None):
        """
        Args:
            src: [batch_size x pkt_num x seq_length]
            tgt: [batch_size]
            seg: [batch_size x pkt_num x seq_length]
            seg_flow: [batch_size x pkt_num]
            seq_attr: [batch_size x pkt_num x attr_dim]
            soft_tgt: [batch_size x labels_num]

            output: [batch_size x pkt_num x hidden_size]
        """
        # Embedding.
        emb = self.embedding(src, seg)
        # Encoder1.
        output1 = self.encoder1(emb, seg)
        temp_output = output1
        # Target1.
        if self.pooling1 == "mean":
            output1 = torch.mean(output1, dim=-2)
        elif self.pooling1 == "max":
            output1 = torch.max(output1, dim=-2)[0]
        elif self.pooling1 == "last":
            output1 = output1[:, :, -1, :]
        else:
            output1 = output1[:, :, 0, :]

        # 将payload输出与attr特征拼接起来
        if seq_attr is not None:
            output1 = torch.cat((output1, seq_attr), dim=-1)
            output1 = self.attr_embedding(output1)

        # Encoder2.
        output2 = self.encoder2(output1, seg_flow)
        # Target2.
        if self.pooling2 == "mean":
            output2 = torch.mean(output2, dim=-2)
        elif self.pooling2 == "max":
            output2 = torch.max(output2, dim=-2)[0]
        elif self.pooling2 == "last":
            output2 = output2[:, -1, :]
        else:
            output2 = output2[:, 0, :]
        
        output2 = torch.tanh(self.output_layer_1(output2))
        logits = self.output_layer_2(output2)
        if tgt is not None:
            if self.soft_targets and soft_tgt is not None:
                loss = self.soft_alpha * nn.MSELoss()(logits, soft_tgt) + \
                       (1 - self.soft_alpha) * nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            else:
                loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            return loss, logits
        else:
            return None, logits
            #return temp_output, logits


def count_labels_num(path):
    labels_set, columns = set(), {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line.strip().split("\t")
            label = int(line[columns["label"]])
            labels_set.add(label)
    return len(labels_set)


def load_or_initialize_parameters(args, model):
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model.load_state_dict(torch.load(args.pretrained_model_path, map_location={'cuda:1':'cuda:0', 'cuda:2':'cuda:0', 'cuda:3':'cuda:0'}), strict=False)
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)


def build_optimizer(args, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    if args.optimizer in ["adamw"]:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    else:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                  scale_parameter=False, relative_step=False)
    if args.scheduler in ["constant"]:
        scheduler = str2scheduler[args.scheduler](optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup)
    else:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup, args.train_steps)
    return optimizer, scheduler


def batch_loader(batch_size, src, tgt, seg, seg_flow, seq_attr=None, soft_tgt=None):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size : (i + 1) * batch_size, :, :]
        tgt_batch = tgt[i * batch_size : (i + 1) * batch_size]
        seg_batch = seg[i * batch_size : (i + 1) * batch_size, :, :]
        seg_flow_batch = seg_flow[i * batch_size : (i + 1) * batch_size, :]
        if seq_attr is not None:
            seq_attr_batch = seq_attr[i * batch_size : (i + 1) * batch_size, :, :]
            if soft_tgt is not None:
                soft_tgt_batch = soft_tgt[i * batch_size : (i + 1) * batch_size, :]
                yield src_batch, tgt_batch, seg_batch, seg_flow_batch, seq_attr_batch, soft_tgt_batch
            else:
                yield src_batch, tgt_batch, seg_batch, seg_flow_batch, seq_attr_batch, None
        else:
            if soft_tgt is not None:
                soft_tgt_batch = soft_tgt[i * batch_size : (i + 1) * batch_size, :]
                yield src_batch, tgt_batch, seg_batch, seg_flow_batch, None, soft_tgt_batch
            else:
                yield src_batch, tgt_batch, seg_batch, seg_flow_batch, None, None
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size :, :, :]
        tgt_batch = tgt[instances_num // batch_size * batch_size :]
        seg_batch = seg[instances_num // batch_size * batch_size :, :, :]
        seg_flow_batch = seg_flow[instances_num // batch_size * batch_size :, :]
        if seq_attr is not None:
            seq_attr_batch = seq_attr[instances_num // batch_size * batch_size :, :, :]
            if soft_tgt is not None:
                soft_tgt_batch = soft_tgt[instances_num // batch_size * batch_size :, :]
                yield src_batch, tgt_batch, seg_batch, seg_flow_batch, seq_attr_batch, soft_tgt_batch
            else:
                yield src_batch, tgt_batch, seg_batch, seg_flow_batch, seq_attr_batch, None
        else:
            if soft_tgt is not None:
                soft_tgt_batch = soft_tgt[instances_num // batch_size * batch_size :, :]
                yield src_batch, tgt_batch, seg_batch, seg_flow_batch, None, soft_tgt_batch
            else:
                yield src_batch, tgt_batch, seg_batch, seg_flow_batch, None, None


def read_dataset(args, path):
    dataset, columns = [], {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            # 删去line的-1相当于strip
            line = line[:-1].split("\t")
            # 第一列为label
            tgt = int(line[columns["label"]])
            if args.soft_targets and "logits" in columns.keys():
                soft_tgt = [float(value) for value in line[columns["logits"]].split(" ")]
            if "text_b" not in columns:  # Sentence classification.
                text_a = line[columns["text_a"]]
                src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a))
                seg = [1] * len(src)
            else:  # Sentence-pair classification.
                text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]
                src_a = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN])
                src_b = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(text_b) + [SEP_TOKEN])
                src = src_a + src_b
                seg = [1] * len(src_a) + [2] * len(src_b)

            if len(src) > args.seq_length:
                src = src[: args.seq_length]
                seg = seg[: args.seq_length]
            while len(src) < args.seq_length:
                src.append(0)
                seg.append(0)
            if args.soft_targets and "logits" in columns.keys():
                dataset.append((src, tgt, seg, soft_tgt))
            else:
                dataset.append((src, tgt, seg))

    return dataset

# 包含除payload之外的其他特征
def read_dataset_with_other_features(args, path):
    dataset, columns = [], {}
    with open(path, mode="r", encoding="utf-8") as f:
        seq_payload = []
        seq_attr = []
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            # 删去line的-1相当于strip
            line = line[:-1].split("\t")
            # 第一列为label
            tgt = int(line[columns["label"]])
            seq_payload = line[columns["payload"]].split('|')
            if args.soft_targets and "logits" in columns.keys():
                soft_tgt = [float(value) for value in line[columns["logits"]].split(" ")]
            # 有包的字节序列长度和包个数两个截断
            # 包维度的seg
            seg_flow = [1] * len(seq_payload)
            seg_flow = seg_flow[: args.pkt_num] + [0] * (args.pkt_num - len(seq_payload))
            # 将payload转化为id序列
            src = [args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(seq_payload[i])) for i in range(len(seq_payload))]
            # 截断和padding包个数
            src = src[: args.pkt_num] + [args.tokenizer.convert_tokens_to_ids([CLS_TOKEN])] * (args.pkt_num - len(src))
            # 包内payload的seg
            seg = [[1] * len(src[i]) for i in range(len(src))]
            # 截断和padding包内payload长度和seg
            src = [src[i][: args.seq_length] + [0] * (args.seq_length - len(src[i])) for i in range(len(src))]
            seg = [seg[i][: args.seq_length] + [0] * (args.seq_length - len(seg[i])) for i in range(len(seg))]
            # 将剩下的列，以|分割，并将形状变为（包数据×特征维度）
            if args.attr:
                attr = [line[i].split('|') for i in range(len(line)) if i != columns["label"] and i != columns["payload"]]
                seq_attr = [[float(row[i]) for row in attr] for i in range(len(attr[0]))][: args.pkt_num] + [[0] * len(attr)] * (args.pkt_num - len(attr[0]))
                if args.soft_targets and "logits" in columns.keys():
                    dataset.append((src, tgt, seg, seg_flow, seq_attr, soft_tgt))
                else:
                    dataset.append((src, tgt, seg, seg_flow, seq_attr))
            else:
                if args.soft_targets and "logits" in columns.keys():
                    dataset.append((src, tgt, seg, seg_flow, None, soft_tgt))
                else:
                    dataset.append((src, tgt, seg, seg_flow, None))
    if args.attr:
        feas_type = {}
        feas_type['num_feas'] = [columns[col]-2 for col in columns.keys() if col in {'length', 'time', 'delta_time'}]
        feas_type['cate_feas'] = [columns[col]-2 for col in columns.keys() if col in {'syn', 'ack', 'fin', 'rst', 'psh', 'urg'}]
        return dataset, feas_type

    return dataset


def process_dataset(args, dataset, setname, feas_type=None, min_value=None, max_value=None):
    src = torch.LongTensor([example[0] for example in dataset])
    tgt = torch.LongTensor([example[1] for example in dataset])
    seg = torch.LongTensor([example[2] for example in dataset])
    seg_flow = torch.LongTensor([example[3] for example in dataset])
    if args.attr:
        seq_attr = torch.FloatTensor([example[4] for example in dataset])
        if 'num_feas' in feas_type:
            if min_value is None or max_value is None:
                min_value = torch.min(torch.min(seq_attr[:, :, feas_type['num_feas']], dim=-2)[0], dim=-2)[0]
                max_value = torch.max(torch.max(seq_attr[:, :, feas_type['num_feas']], dim=-2)[0], dim=-2)[0]
            seq_attr[:, :, feas_type['num_feas']] = (seq_attr[:, :, feas_type['num_feas']] - min_value) / (max_value - min_value)
        if 'cate_feas' in feas_type:
            pass
    else:
        seq_attr = None
    if args.soft_targets:
        soft_tgt = torch.FloatTensor([example[5] for example in dataset])
    else:
        soft_tgt = None

    return src, tgt, seg, seg_flow, seq_attr, soft_tgt, min_value, max_value


def train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, seg_flow_batch, seq_attr_batch=None, soft_tgt_batch=None):
    model.zero_grad()

    src_batch = src_batch.to(args.device)
    tgt_batch = tgt_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)
    seg_flow_batch = seg_flow_batch.to(args.device)
    seq_attr_batch = seq_attr_batch.to(args.device)
    if soft_tgt_batch is not None:
        soft_tgt_batch = soft_tgt_batch.to(args.device)

    loss, _ = model(src_batch, tgt_batch, seg_batch, seg_flow_batch, seq_attr_batch, soft_tgt_batch)
    if torch.cuda.device_count() > 1:
        loss = torch.mean(loss)

    if args.fp16:
        with args.amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    optimizer.step()
    scheduler.step()

    return loss


def evaluate(args, model, src, tgt, seg, seg_flow, seq_attr, soft_tgt, setname, print_confusion_matrix=False):
    print(f"Evaluating {setname} dataset:")

    batch_size = args.batch_size

    correct = 0
    # Confusion matrix.
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

    model.eval()

    for i, (src_batch, tgt_batch, seg_batch, seg_flow_batch, seq_attr_batch, soft_tgt_batch) in enumerate(batch_loader(batch_size, src, tgt, seg, seg_flow, seq_attr, soft_tgt)):
        src_batch = src_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        seg_flow_batch = seg_flow_batch.to(args.device)
        seq_attr_batch = seq_attr_batch.to(args.device)
        if soft_tgt_batch is not None:
            soft_tgt_batch = soft_tgt_batch.to(args.device)

        with torch.no_grad():
            _, logits = model(src_batch, tgt_batch, seg_batch, seg_flow_batch, seq_attr_batch, soft_tgt_batch)
        pred = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
        gold = tgt_batch
        for j in range(pred.size()[0]):
            confusion[pred[j], gold[j]] += 1
        correct += torch.sum(pred == gold).item()

    if print_confusion_matrix:
        print("Confusion matrix:")
        print(confusion)
        cf_array = confusion.numpy()
        with open("fine_tuning/results/confusion_matrix", 'w') as f:
            for cf_a in cf_array:
                f.write(str(cf_a)+'\n')
        print("Report precision, recall, and f1:")
        if not args.label_id_path:
            print("No label_id_path provided, using default label categories.")
            label_cat = {i: i for i in range(args.labels_num)}
        else:
            with open(args.label_id_path) as js:
                label_id = json.load(js)
                label_cat = dict(zip(label_id.values(), label_id.keys()))
        eps = 1e-9
        for i in range(confusion.size()[0]):
            p = confusion[i, i].item() / (confusion[i, :].sum().item() + eps)
            r = confusion[i, i].item() / (confusion[:, i].sum().item() + eps)
            if (p + r) == 0:
                f1 = 0
            else:
                f1 = 2 * p * r / (p + r)
            print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(label_cat[i], p, r, f1))

    print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct / len(tgt), correct, len(tgt)))
    return correct / len(tgt), confusion


def main():
    print("Starting\n")
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    finetune_opts(parser)
    
    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)
    # Print the arguments.
    print("Arguments:\n")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
        
    print("\nCounting labels.\n")
    # Count the number of labels.
    args.labels_num = count_labels_num(args.train_path)
    print("Building tokenizer.\n")
    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)
    print("Building model.\n")
    # Build classification model.
    model = Classifier(args)
    # print(model)
    # params = model.state_dict() 
    # for k,v in params.items():
    #     if 'weight' in k and 'encoder' in k:
    #         print(k, v)
    #         break
    # args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    # 先改变模型，添加lora，防止后添加lora原参数被覆盖
    if args.lora_r:
        model = get_lora_bert_model(model, args.lora_r, ['q', 'v', 'k', 'o'])
        lora.mark_only_lora_as_trainable(model)
    # Load or initialize parameters.
    load_or_initialize_parameters(args, model)

    # model = model.to(args.device)
    # evaluate(args, model, read_dataset(args, args.dev_path), 'valid', True)
    # # params = model.state_dict() 
    # # for k,v in params.items():
    # #     print(k)
    # save_model(model, args.output_model_path, args.lora_r)
    # # model_dict = merge_model(model)
    # # for k,v in model_dict.items():
    # #     if 'weight' in k and 'encoder' in k:
    # #         print(k, v)
    # #         break
    # model_new = Classifier(args)
    # model_new = model_new.to(args.device)
    # model_new.load_state_dict(torch.load('models/lora_checkpoint/lora_model_checkpoint9.bin'))
    # model_dict = model_new.state_dict()
    # for k,v in model_dict.items():
    #     if 'weight' in k and 'encoder' in k:
    #         print(k, v)
    #         break

    # evaluate(args, model_new, read_dataset(args, args.test_path), 'test', True)

    
    print(model)
    print([n for n, p in model.named_parameters() if 'lora' in n])
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)
    print("Reading dataset.\n")
    # Training phase.
    # 是否使用除包payload之外的其他特征
    # if not args.attr:
    #     trainset = read_dataset(args, args.train_path)
    # else:
    trainset, feas_type = read_dataset_with_other_features(args, args.train_path)
    random.shuffle(trainset)

    # 解析dataset，获取min_value和max_value
    src, tgt, seg, seg_flow, seq_attr, soft_tgt, min_value, max_value = process_dataset(args, trainset, 'train', feas_type)

    # 获取验证集和测试集数据
    devset, _ = read_dataset_with_other_features(args, args.dev_path)
    testset, _ = read_dataset_with_other_features(args, args.test_path)
    dev_src, dev_tgt, dev_seg, dev_seg_flow, dev_seq_attr, dev_soft_tgt, _, _ = process_dataset(args, devset, 'valid', feas_type, min_value, max_value)
    test_src, test_tgt, test_seg, test_seg_flow, test_seq_attr, test_soft_tgt, _, _ = process_dataset(args, testset, 'test', feas_type, min_value, max_value)

    instances_num = len(trainset)
    batch_size = args.batch_size
    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    print("Batch size: ", batch_size)
    print("The number of training instances:", instances_num)

    optimizer, scheduler = build_optimizer(args, model)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        args.amp = amp

    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    args.model = model

    total_loss, result, best_result = 0.0, 0.0, 0.0

    print("Start training.")

    for epoch in tqdm.tqdm(range(1, args.epochs_num + 1)):
        model.train()
        for i, (src_batch, tgt_batch, seg_batch, seg_flow_batch, seq_attr_batch, soft_tgt_batch) in enumerate(batch_loader(batch_size, src, tgt, seg, seg_flow, seq_attr, soft_tgt)):
            loss = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, seg_flow_batch, seq_attr_batch, soft_tgt_batch)
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1, total_loss / args.report_steps))
                total_loss = 0.0
                merge_model(model)
        # if args.lora_r:
        #     checkpoint_path = f"models/lora_checkpoint/lora_model_checkpoint{epoch}.bin"
        #     save_model(model, checkpoint_path, args.lora_r)

        #     model_eval = Classifier(args)
        #     model_eval = get_lora_bert_model(model_eval, args.lora_r, ['q', 'v'])
        #     model_eval.load_state_dict(torch.load(args.pretrained_model_path), strict=False)
        #     model_eval.load_state_dict(torch.load(checkpoint_path), strict=False)
        #     model_eval = model_eval.to(args.device)
        # evaluate(args, model, trainset, 'train')
        # evaluate(args, model, read_dataset(args, args.dev_path), 'valid')
        train_res = evaluate(args, model, src, tgt, seg, seg_flow, seq_attr, soft_tgt, 'train')
        result = evaluate(args, model, dev_src, dev_tgt, dev_seg, dev_seg_flow, dev_seq_attr, dev_soft_tgt, 'valid')
        test_res = evaluate(args, model, test_src, test_tgt, test_seg, test_seg_flow, test_seq_attr, test_soft_tgt, 'test', True)
        if args.lora_r:
            checkpoint_path = f"models/lora_checkpoint/lora_model_checkpoint{epoch}.bin"
            save_model(model, checkpoint_path, args.lora_r)
        if result[0] > best_result:
            print("Better result and save the model.")
            best_result = result[0]
            save_model(model, args.output_model_path, args.lora_r)

    # Evaluation phase.
    if args.test_path is not None:
        model_eval = Classifier(args)
        model_eval = model_eval.to(args.device)
        print("Test set evaluation.")
        # if args.lora_r:
        #     if torch.cuda.device_count() > 1:
        #         model.module.load_state_dict(torch.load(args.pretrained_model_path), strict=False)
        #         model.module.load_state_dict(torch.load(args.output_model_path), strict=False)
        #     else:
        #         model.load_state_dict(torch.load(args.pretrained_model_path), strict=False)
        #         model.load_state_dict(torch.load(args.output_model_path), strict=False)
        # else:
        if torch.cuda.device_count() > 1:
            model_eval.module.load_state_dict(torch.load(args.output_model_path))
        else:
            model_eval.load_state_dict(torch.load(args.output_model_path))
        evaluate(args, model_eval, test_src, test_tgt, test_seg, test_seg_flow, test_seq_attr, test_soft_tgt, 'test', True)
    print('Already finished!')


if __name__ == "__main__":
    main()

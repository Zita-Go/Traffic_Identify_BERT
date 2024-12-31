#!/usr/bin/python3
#-*- coding:utf-8 -*-

import numpy as np
import json
import os
import time
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
from scipy.stats import skew,kurtosis
import sys
import csv
import copy
import tqdm
import random
import shutil
import dataset_generation

import data_preprocess
import open_dataset_deal

_category = 7 # dataset class
# 此目录用于存放微调用的数据集。其与dataset_save_path不同，dataset_save_path分开存放特征和标签，且格式为npy文件，而此目录存放的是合并后的特征和标签，格式为tsv文件。
# dataset_dir = "I:\\datasets\\" # the path to save dataset for dine-tuning


# 分别表示，原始数据地址、处理后数据存放地址、所有类别的最大样本数、特征选取、流量识别维度
# pcap_path, dataset_save_path, samples, features, dataset_level = "I:\\cstnet-tls1.3\\packet\\splitcap\\", "I:\\cstnet-tls1.3\\packet\\result\\", [5000], ["payload"], "packet"

samples = [5000]
features = ["payload", "length", "time", "delta_time", "syn", "ack", "fin", "rst", "psh", "urg"]
dataset_level = "burst"
dataset_dir = f"datasets\\blockchain_traffic\\{dataset_level}\\"
pcap_path = f"datasets\\blockchain_traffic\\{dataset_level}\\splitcap\\"
dataset_save_path = f"datasets\\blockchain_traffic\\{dataset_level}\\processed_data\\"

def dataset_extract(model):
    
    X_dataset = {}
    Y_dataset = {}

    if not os.path.exists(dataset_save_path):
        os.mkdir(dataset_save_path)

    try:
        if os.listdir(dataset_save_path + "dataset\\"):
            print("Reading dataset from %s ..." % (dataset_save_path + "dataset\\"))
            
            x_payload_train, x_payload_test, x_payload_valid,\
            y_train, y_test, y_valid = \
                np.load(dataset_save_path + "dataset\\x_datagram_train.npy",allow_pickle=True), np.load(dataset_save_path + "dataset\\x_datagram_test.npy",allow_pickle=True), np.load(dataset_save_path + "dataset\\x_datagram_valid.npy",allow_pickle=True),\
                np.load(dataset_save_path + "dataset\\y_train.npy",allow_pickle=True), np.load(dataset_save_path + "dataset\\y_test.npy",allow_pickle=True), np.load(dataset_save_path + "dataset\\y_valid.npy",allow_pickle=True)
            
            X_dataset, Y_dataset = models_deal(model, X_dataset, Y_dataset,
                                               x_payload_train, x_payload_test,
                                               x_payload_valid,
                                               y_train, y_test, y_valid)

            return X_dataset, Y_dataset
    except Exception as e:
        print(e)
        # 没有数据集则需要生成
        print("Dataset directory %s not exist.\nBegin to obtain new dataset."%(dataset_save_path + "dataset\\"))

    # 获得特征和标签，都为列表格式，每个特征构成一个列表
    X,Y = dataset_generation.generation(pcap_path, samples, features, splitcap=False, dataset_save_path=dataset_save_path,dataset_level=dataset_level)

    dataset_statistic = [0] * _category

    # 不再按照特征和标签分组，而是按照样本随机，X_payload为所有样本的payload，Y_all为所有样本的标签
    X_payload= []
    Y_all = []
    # for app_label in Y:
    #     for label in app_label:
    #         Y_all.append(int(label))
    # 统计每个类别的个数
    for label_id in range(_category):
        for label in Y:
            if int(label) == label_id:
                dataset_statistic[label_id] += 1
    print("category flow")
    for index in range(len(dataset_statistic)):
        print("%s\t%d" % (index, dataset_statistic[index]))
    print("all\t%d" % (sum(dataset_statistic)))

    x_payload = np.array(X)
    dataset_label = np.array(Y)
    # if len(features) == 1:
    #     X_payload = X[0]
    #     x_payload = np.array(X_payload)
    #     dataset_label = np.array(Y_all)
    # elif len(features) > 1:
    #     X_payload = X
    #     x_payload = np.array(X_payload).T
    #     dataset_label = np.array(Y_all)
    # else:
    #     raise ValueError("No feature selected.")

    # for i in range(len(features)):
    #     if features[i] == "payload":
    #         for index_label in range(len(X[0])):
    #             for index_sample in range(len(X[0][index_label])):
    #                 X_payload.append(X[0][index_label][index_sample])

    split_1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=41) 
    split_2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42) 

    x_payload_train = []
    y_train = []

    x_payload_valid = []
    y_valid = []

    x_payload_test = []
    y_test = []

    if dataset_level == "packet":
        for train_index, test_index in split_1.split(x_payload, dataset_label):
            x_payload_train, y_train = x_payload[train_index], dataset_label[train_index]
            x_payload_test, y_test = x_payload[test_index], dataset_label[test_index]
        for test_index, valid_index in split_2.split(x_payload_test, y_test):
            x_payload_valid, y_valid = x_payload_test[valid_index], y_test[valid_index]
            x_payload_test, y_test = x_payload_test[test_index], y_test[test_index]
    else:
        for train_index, test_index in split_1.split(x_payload, dataset_label):
            x_payload_train, y_train = \
                x_payload[train_index], \
                dataset_label[train_index]
            x_payload_test,y_test = \
                x_payload[test_index], \
                dataset_label[test_index]
        for test_index, valid_index in split_2.split(x_payload_test, y_test):
            x_payload_valid, y_valid = \
                x_payload_test[valid_index], y_test[valid_index]
            x_payload_test, y_test = \
                x_payload_test[test_index], y_test[test_index]

    if not os.path.exists(dataset_save_path+"dataset\\"):
        os.mkdir(dataset_save_path+"dataset\\")

    output_x_payload_train = os.path.join(dataset_save_path + "dataset\\", 'x_datagram_train.npy')

    output_x_payload_test = os.path.join(dataset_save_path + "dataset\\", 'x_datagram_test.npy')

    output_x_payload_valid = os.path.join(dataset_save_path + "dataset\\", 'x_datagram_valid.npy')

    output_y_train = os.path.join(dataset_save_path+"dataset\\",'y_train.npy')
    output_y_test = os.path.join(dataset_save_path + "dataset\\", 'y_test.npy')
    output_y_valid = os.path.join(dataset_save_path + "dataset\\", 'y_valid.npy')

    np.save(output_x_payload_train, x_payload_train)
    np.save(output_x_payload_test, x_payload_test)
    np.save(output_x_payload_valid, x_payload_valid)

    np.save(output_y_train, y_train)
    np.save(output_y_test, y_test)
    np.save(output_y_valid, y_valid)

    X_dataset, Y_dataset = models_deal(model, X_dataset, Y_dataset,
                                       x_payload_train, x_payload_test, x_payload_valid,
                                       y_train, y_test, y_valid)

    return X_dataset,Y_dataset

def models_deal(model, X_dataset, Y_dataset, x_payload_train, x_payload_test, x_payload_valid, y_train, y_test, y_valid):
    for index in range(len(model)):
        print("Begin to model %s dealing..."%model[index])
        x_train_dataset = []
        x_test_dataset = []
        x_valid_dataset = []

        if model[index] == "pre-train":
            save_dir = dataset_dir
            write_dataset_tsv(x_payload_train, y_train, save_dir, "train")
            write_dataset_tsv(x_payload_test, y_test, save_dir, "test")
            write_dataset_tsv(x_payload_valid, y_valid, save_dir, "valid")
            print("finish generating pre-train's datagram dataset.\nPlease check in %s" % save_dir)
            unlabel_data(dataset_dir + "test_dataset.tsv")

        X_dataset[model[index]] = {"train": [], "valid": [], "test": []}
        Y_dataset[model[index]] = {"train": [], "valid": [], "test": []}

        X_dataset[model[index]]["train"], Y_dataset[model[index]]["train"] = x_train_dataset, y_train
        X_dataset[model[index]]["valid"], Y_dataset[model[index]]["valid"] = x_valid_dataset, y_valid
        X_dataset[model[index]]["test"], Y_dataset[model[index]]["test"] = x_test_dataset, y_test

    return X_dataset, Y_dataset

# 将数据存入tsv，格式为标签和payload
def write_dataset_tsv(data,label,file_dir,type):
    # 要保存的特征列名
    with open(dataset_save_path + "\\dataset.json", "r") as f:
        new_dataset = json.load(f)
    dataset_file = [["label"] + [key for key in new_dataset[list(new_dataset.keys())[0]].keys() if key != "samples"]]
    # dataset_file = [["label", "text_a"]]
    if data.shape[1] != len(dataset_file[0])-1:
        raise ValueError("The number of features is not equal to the number of columns in the dataset.json file.")
    for index in range(len(label)):
        # 判断array维度
        if len(data[index]) == 1:
            dataset_file.append([label[index], data[index]])
        elif len(data[index]) > 1:
            dataset_file.append([label[index]] + list(data[index]))
        else:
            raise ValueError("No data found.")
    with open(file_dir + type + "_dataset.tsv", 'w',newline='') as f:
        tsv_w = csv.writer(f, delimiter='\t')
        tsv_w.writerows(dataset_file)
    return 0

def unlabel_data(label_data):
    nolabel_data = ""
    with open(label_data,newline='') as f:
        data = csv.reader(f,delimiter='\t')
        for row in data:
            nolabel_data += row[1] + '\n'
    nolabel_file = label_data.replace("test_dataset","nolabel_test_dataset")
    #nolabel_file = label_data.replace("train_dataset", "nolabel_train_dataset")
    with open(nolabel_file, 'w',newline='') as f:
        f.write(nolabel_data)
    return 0

def cut_byte(obj, sec):
    result = [obj[i:i+sec] for i in range(0,len(obj),sec)]
    remanent_count = len(result[0])%2
    if remanent_count == 0:
        pass
    else:
        result = [obj[i:i+sec+remanent_count] for i in range(0,len(obj),sec+remanent_count)]
    return result

def pickle_save_data(path_file, data):
    with open(path_file, "wb") as f:
        pickle.dump(data, f)
    return 0

def count_label_number(samples):
    new_samples = samples * _category
    
    if 'splitcap' not in pcap_path:
        dataset_length, labels = open_dataset_deal.statistic_dataset_sample_count(pcap_path + 'splitcap\\')
    else:
        dataset_length, labels = open_dataset_deal.statistic_dataset_sample_count(pcap_path)

    for index in range(len(dataset_length)):
        if dataset_length[index] < samples[0]:
            print("label %s has less sample's number than defined samples %d" % (labels[index], samples[0]))
            new_samples[index] = dataset_length[index]
    return new_samples


if __name__ == '__main__':
    open_dataset_not_pcap = 0
    
    # 将其他文件格式转换为pcap，比如旧的pcapng格式
    if open_dataset_not_pcap:
        #open_dataset_deal.dataset_file2dir(pcap_path)
        for p,d,f in os.walk(pcap_path):
            for file in f:
                target_file = file.replace('.','_new.')
                open_dataset_deal.file_2_pcap(p+"\\"+file, p+"\\"+target_file)
                if '_new.pcap' not in file:
                    os.remove(p+"\\"+file)

    # 将每个pcap放入对应的类别的文件夹中
    file2dir = 0
    if file2dir:
        open_dataset_deal.dataset_file2dir(pcap_path)

    splitcap_finish = 1
    # samples是一个列表，表示所有类别流量的样本数
    if splitcap_finish:
        samples = count_label_number(samples)
    else:
        samples = samples * _category

    train_model = ["pre-train"]
    ml_experiment = 0

    dataset_extract(train_model)

#!/usr/bin/python3
#-*- coding:utf-8 -*-

import scapy.all as scapy
import binascii
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
import json
import os
import csv
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from flowcontainer.extractor import extract
import tqdm
import random

random.seed(40)

# pcap_dir = "I:\\dataset\\"
pcap_dir = 'datasets\\blockchain_traffic\\pcap\\'
tls_date = [20210301,20210808]
pcap_name = "app_A.pcap"
#pcap_name = "merge.pcap"
pcap_flag = '.pcap'

# word_dir = "I:/corpora/"
# word_name = "encrypted_tls13_burst.txt"
# vocab_dir = "I:/models/"
# vocab_name = "encryptd_vocab_all.txt"

# 区块链流量数据集
word_dir = 'corpora\\'
word_name = 'blockchain_traffic_data.txt'
vocab_dir = 'models\\'
vocab_name = "blockchain_traffic_vocab_all.txt"
dataset_level = 'burst'
splitpcap_path = f"datasets\\blockchain_traffic\\{dataset_level}\\splitcap\\"


# 将多个时间的pcap包文件夹处理并拼接起来
def pcap_preprocess():
    
    start_date = tls_date[0]
    end_date = tls_date[1]
    packet_num = 0
    while start_date <= end_date:
        data_dir = tls13_pcap_dir + str(start_date) + "\\"
        p_num = preprocess(data_dir)
        packet_num += p_num
        start_date += 1
    print("used packets %d"%packet_num)
    print("finish generating tls13 pretrain dataset.\n please check in %s"%word_dir)
    return 0


# 单个pcap包处理，被pcap_preprocess调用
def preprocess(pcap_dir, dataset_level='packet'):
    print("now pre-process pcap_dir is %s"%pcap_dir)
    
    packet_num = 0
    n = 0

    # 处理一个文件夹中的多个pcap包
    for parent,dirs,files in os.walk(pcap_dir):
        for file in files:
            # pcapng格式是老pcap格式
            if "pcapng" not in file and pcap_flag in file:
                n += 1
                pcap_name = parent + "\\" + file
                print("No.%d pacp is processed ... %s ..."%(n,file))

                # 一次性将所有包读取到内存中，成为一个列表，建议采用scapy.PcapReader
                packets = scapy.rdpcap(pcap_name)
                #word_packet = b''
                words_txt = []

                for p in packets:
                    packet_num += 1
                    word_packet = p.copy()
                    # bytes将字符串变为二进制，hexlify将二进制变为十六进制的字节字符串
                    words = (binascii.hexlify(bytes(word_packet)))

                    # 跳过数据包头，decode是将字节字符串转变为十六进制的普通字符串，每个字符为4bit
                    words_string = words.decode()[76:]
                    # print(words_string)
                    length = len(words_string)
                    if length < 10:
                        continue
                    # payload分为两段
                    for string_txt in cut(words_string, int(length / 2)):
                        token_count = 0
                        # sentence为每两个一对
                        sentence = cut(string_txt,1)  
                        for sub_string_index in range(len(sentence)):
                            if sub_string_index != (len(sentence) - 1):
                                token_count += 1
                                # token限制长度为256，是否还可以增加
                                if token_count > 256:
                                    break
                                else:
                                    # 每个token之间都重复2字节，为4字节
                                    merge_word_bigram = sentence[sub_string_index] + sentence[
                                                                 sub_string_index + 1]  
                            else:
                                break  
                            words_txt.append(merge_word_bigram)
                            words_txt.append(' ')
                        words_txt.append("\n")
                    words_txt.append("\n")

                
                result_file = open(word_dir + word_name, 'a')
                for words in words_txt:
                    result_file.write(words)
                result_file.close()

                split_cap(pcap_name, splitpcap_path, dataset_level)
    print("finish preprocessing %d pcaps"%n)
    return packet_num

def cut(obj, sec):
    result = [obj[i:i+sec] for i in range(0,len(obj),sec)]
    remanent_count = len(result[0])%4
    if remanent_count == 0:
        pass
    else:
        # 保证每一个是偶数字节的
        result = [obj[i:i+sec+remanent_count] for i in range(0,len(obj),sec+remanent_count)]
    return result


# BPE为Byte Pair Encoding，不断聚合高频词，得到token
def build_BPE():
    # generate source dictionary,0-65535
    num_count = 65536
    not_change_string_count = 5
    i = 0
    source_dictionary = {} 
    tuple_sep = ()
    tuple_cls = ()
    #'PAD':0,'UNK':1,'CLS':2,'SEP':3,'MASK':4
    while i < num_count:
        # 4位数16进制，不够补零
        temp_string = '{:04x}'.format(i) 
        source_dictionary[temp_string] = i
        i += 1

    # 使用wordpiece获得token词表
    # Initialize a tokenizer
    tokenizer = Tokenizer(models.WordPiece(vocab=source_dictionary,unk_token="[UNK]",max_input_chars_per_word=4))

    # Customize pre-tokenization and decoding
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.decoder = decoders.WordPiece()
    tokenizer.post_processor = processors.BertProcessing(sep=("[SEP]",1),cls=('[CLS]',2))

    # And then train
    trainer = trainers.WordPieceTrainer(vocab_size=65536, min_frequency=2)
    tokenizer.train([word_dir+word_name, word_dir+word_name], trainer=trainer)

    # And Save it
    tokenizer.save("wordpiece.tokenizer.json", pretty=True)
    return 0


# 直接从tokenizer中获取token及其对应的index
def build_vocab():
    json_file = open("wordpiece.tokenizer.json",'r')
    json_content = json_file.read()
    json_file.close()
    vocab_json = json.loads(json_content)
    vocab_txt = ["[PAD]","[SEP]","[CLS]","[UNK]","[MASK]"]
    for item in vocab_json['model']['vocab']:
        vocab_txt.append(item) # append key of vocab_json
    with open(vocab_dir+vocab_name,'w') as f:
        for word in vocab_txt:
            f.write(word+"\n")
    return 0

def bigram_generation(packet_string,flag=False):
    result = ''
    sentence = cut(packet_string,1)
    token_count = 0
    for sub_string_index in range(len(sentence)):
        if sub_string_index != (len(sentence) - 1):
            token_count += 1
            if token_count > 256: 
                break
            else:
                merge_word_bigram = sentence[sub_string_index] + sentence[sub_string_index + 1]
        else:
            break
        result += merge_word_bigram
        result += ' '
    if flag == True:
        result = result.rstrip()

    return result

def read_pcap_feature(pcap_file):
    packet_length_feature = []
    feature_result = extract(pcap_file, filter='tcp')
    for key in feature_result.keys():
        value = feature_result[key]
        packet_length_feature.append(value.ip_lengths)
    return packet_length_feature[0]

def read_pcap_flow(pcap_file):
    packets = scapy.rdpcap(pcap_file)

    packet_count = 0  
    flow_data_string = '' 

    if len(packets) < 5:
        print("preprocess flow %s but this flow has less than 5 packets."%pcap_file)
        return -1

    print("preprocess flow %s" % pcap_file)
    for packet in packets:
        packet_count += 1
        if packet_count == 5:
            packet_data = packet.copy()
            data = (binascii.hexlify(bytes(packet_data)))
            packet_string = data.decode()
            flow_data_string += bigram_generation(packet_string,flag = True)
            break
        else:
            packet_data = packet.copy()
            data = (binascii.hexlify(bytes(packet_data)))
            packet_string = data.decode()
            flow_data_string += bigram_generation(packet_string)
    return flow_data_string

def split_cap(pcap_file, pcap_path, dataset_level = 'packet'):
    if not os.path.exists(pcap_path):
        os.makedirs(pcap_path)
    if dataset_level == 'packet':
        cmd = "D:/Users/ZitaGo/Downloads/SplitCap.exe -r %s -s packets 1 -o " + pcap_path
    elif dataset_level =='flow' or dataset_level == 'burst':
        cmd = "D:/Users/ZitaGo/Downloads/SplitCap.exe -r %s -s session -o " + pcap_path
    command = cmd%pcap_file
    os.system(command)
    return 0

if __name__ == '__main__':
    preprocess(pcap_dir, dataset_level)

    # build vocab
    build_BPE()
    build_vocab()

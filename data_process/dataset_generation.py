#!/usr/bin/python3
#-*- coding:utf-8 -*-

import os
import sys
import copy
import json
import tqdm
import shutil
import pickle
import random
import binascii
import operator
import numpy as np
import pandas as pd
import scapy.all as scapy
from functools import reduce
from flowcontainer.extractor import extract

random.seed(40)

# word_dir = "I:/corpora/"
# word_name = "encrypted_burst.txt"
word_dir = "corpora/"
word_name = "blockchain_traffic_burst.txt"

def convert_pcapng_2_pcap(pcapng_path, pcapng_file, output_path):
    
    pcap_file = output_path + pcapng_file.replace('pcapng','pcap')
    cmd = "I:\\editcap.exe -F pcap %s %s"
    command = cmd%(pcapng_path+pcapng_file, pcap_file)
    os.system(command)
    return 0

def split_cap(pcap_path, pcap_file, pcap_name, pcap_label='', dataset_level = 'flow'):
    
    if not os.path.exists(pcap_path + "\\splitcap"):
        os.mkdir(pcap_path + "\\splitcap")
    if pcap_label != '':
        if not os.path.exists(pcap_path + "\\splitcap\\" + pcap_label):
            os.mkdir(pcap_path + "\\splitcap\\" + pcap_label)
        if not os.path.exists(pcap_path + "\\splitcap\\" + pcap_label + "\\" + pcap_name):
            os.mkdir(pcap_path + "\\splitcap\\" + pcap_label + "\\" + pcap_name)
   
        output_path = pcap_path + "\\splitcap\\" + pcap_label + "\\" + pcap_name
    else:
        if not os.path.exists(pcap_path + "\\splitcap\\" + pcap_name):
            os.mkdir(pcap_path + "\\splitcap\\" + pcap_name)
        output_path = pcap_path + "\\splitcap\\" + pcap_name
    if dataset_level == 'flow':
        cmd = "I:\\SplitCap.exe -r %s -s session -o " + output_path
    elif dataset_level == 'packet':
        cmd = "I:\\SplitCap.exe -r %s -s packets 1 -o " + output_path
    command = cmd%pcap_file
    os.system(command)
    return output_path

def cut(obj, sec):
    result = [obj[i:i+sec] for i in range(0,len(obj),sec)]
    try:
        remanent_count = len(result[0])%4
    except Exception as e:
        remanent_count = 0
        print("cut datagram error!")
    if remanent_count == 0:
        pass
    else:
        result = [obj[i:i+sec+remanent_count] for i in range(0,len(obj),sec+remanent_count)]
    return result

def bigram_generation(packet_datagram, packet_len = 64, flag=True):
    result = ''
    generated_datagram = cut(packet_datagram,1)
    token_count = 0
    for sub_string_index in range(len(generated_datagram)):
        if sub_string_index != (len(generated_datagram) - 1):
            token_count += 1
            if token_count > packet_len:
                break
            else:
                merge_word_bigram = generated_datagram[sub_string_index] + generated_datagram[sub_string_index + 1]
        else:
            break
        result += merge_word_bigram
        result += ' '
    
    return result

def get_burst_feature(label_pcap, payload_len):
    feature_data = []
    
    packets = scapy.rdpcap(label_pcap)
    
    packet_direction = []
    feature_result = extract(label_pcap)
    for key in feature_result.keys():
        value = feature_result[key]
        packet_direction = [x // abs(x) for x in value.ip_lengths]

    if len(packet_direction) == len(packets):
        
        burst_data_string = ''
        
        burst_txt = ''
        
        for packet_index in range(len(packets)):
            packet_data = packets[packet_index].copy()
            data = (binascii.hexlify(bytes(packet_data)))
            
            packet_string = data.decode()[:2*payload_len]
            
            if packet_index == 0:
                burst_data_string += packet_string
            else:
                # 方向不同，表示burst结束
                if packet_direction[packet_index] != packet_direction[packet_index - 1]:
                    
                    length = len(burst_data_string)
                    # 分成两半，分别生成bigram
                    for string_txt in cut(burst_data_string, int(length / 2)):
                        burst_txt += bigram_generation(string_txt, packet_len=len(string_txt))
                        burst_txt += '\n'
                    burst_txt += '\n'
                    
                    burst_data_string = ''
                
                burst_data_string += packet_string
                if packet_index == len(packets) - 1:
                    
                    length = len(burst_data_string)
                    for string_txt in cut(burst_data_string, int(length / 2)):
                        burst_txt += bigram_generation(string_txt, packet_len=len(string_txt))
                        burst_txt += '\n'
                    burst_txt += '\n'
        
        with open(word_dir + word_name,'a') as f:
            f.write(burst_txt)
    return 0

def get_feature_bursts_with_empty_payload(label_pcap, payload_len):
    pass

def get_feature_bursts(label_pcap, payload_len, payload_pac, samples_num):
    feature_data = []
    
    packets = scapy.rdpcap(label_pcap)

    # 提取bursts特征，包括包方向、包长度（payload长度）、包时间、6个标志位
    feature_result = extract(label_pcap, filter='tcp', extension=['tcp.flags.syn', 'tcp.flags.ack', 'tcp.flags.fin', 'tcp.flags.reset', 'tcp.flags.push', 'tcp.flags.urg', 'data'])
    
    # 提取包信息
    for key in feature_result.keys():
        value = feature_result[key]
        packet_direction = [x // abs(x) for x in value.payload_lengths]
        packet_time = value.payload_timestamps
        packet_length = value.payload_lengths
        packet_syn = value.extension['tcp.flags.syn']
        packet_ack = value.extension['tcp.flags.ack']
        packet_fin = value.extension['tcp.flags.fin']
        packet_reset = value.extension['tcp.flags.reset']
        packet_push = value.extension['tcp.flags.push']
        packet_urg = value.extension['tcp.flags.urg']
        packet_data = value.extension['data']

    bursts_length = []
    bursts_time = []
    bursts_direction = []
    bursts_data = []
    bursts_syn = []
    bursts_ack = []
    bursts_fin = []
    bursts_reset = []
    bursts_push = []
    bursts_urg = []


    data_dict = {}
    
    if len(packet_direction) == len(packets):
        burst_length = []
        burst_time = []
        burst_direction = []
        burst_data = []
        burst_syn = []
        burst_ack = []
        burst_fin = []
        burst_reset = []
        burst_push = []
        burst_urg = []

        # 分割burst
        for packet_index in range(len(packet_direction)):
            if packet_index == 0:
                burst_length.append(packet_length[packet_index])
                burst_time.append(packet_time[packet_index])
                burst_direction.append(packet_direction[packet_index])
                burst_data.append(bigram_generation(packet_data[packet_index], packet_len=payload_len, flag = True))
                burst_syn.append(1 if packet_syn[packet_index] == 'True' else 0)
                burst_ack.append(1 if packet_ack[packet_index] == 'True' else 0)
                burst_fin.append(1 if packet_fin[packet_index] == 'True' else 0)
                burst_reset.append(1 if packet_reset[packet_index] == 'True' else 0)
                burst_push.append(1 if packet_push[packet_index] == 'True' else 0)
                burst_urg.append(1 if packet_urg[packet_index] == 'True' else 0)
            else:
                if packet_direction[packet_index] != packet_direction[packet_index - 1]:
                    bursts_length.append(burst_length.copy())
                    bursts_time.append(burst_time.copy())
                    bursts_direction.append(burst_direction.copy())
                    bursts_data.append(burst_data.copy())
                    bursts_syn.append(burst_syn.copy())
                    bursts_ack.append(burst_ack.copy())
                    bursts_fin.append(burst_fin.copy())
                    bursts_reset.append(burst_reset.copy())
                    bursts_push.append(burst_push.copy())
                    bursts_urg.append(burst_urg.copy())

                    burst_length = []
                    burst_time = []
                    burst_direction = []
                    burst_data = []
                    burst_syn = []
                    burst_ack = []
                    burst_fin = []
                    burst_reset = []
                    burst_push = []
                    burst_urg = []
                
                if len(burst_length) < payload_pac:
                    burst_length.append(packet_length[packet_index])
                    burst_time.append(packet_time[packet_index])
                    burst_direction.append(packet_direction[packet_index])
                    burst_data.append(bigram_generation(packet_data[packet_index], packet_len=payload_len, flag = True))
                    burst_syn.append(1 if packet_syn[packet_index] == 'True' else 0)
                    burst_ack.append(1 if packet_ack[packet_index] == 'True' else 0)
                    burst_fin.append(1 if packet_fin[packet_index] == 'True' else 0)
                    burst_reset.append(1 if packet_reset[packet_index] == 'True' else 0)                    
                    burst_push.append(1 if packet_push[packet_index] == 'True' else 0)
                    burst_urg.append(1 if packet_urg[packet_index] == 'True' else 0)

                if packet_index == len(packet_direction) - 1:
                    bursts_length.append(burst_length.copy())
                    bursts_time.append(burst_time.copy())
                    bursts_direction.append(burst_direction.copy())
                    bursts_data.append(burst_data.copy())                    
                    bursts_syn.append(burst_syn.copy())
                    bursts_ack.append(burst_ack.copy())
                    bursts_fin.append(burst_fin.copy())
                    bursts_reset.append(burst_reset.copy())
                    bursts_push.append(burst_push.copy())
                    bursts_urg.append(burst_urg.copy())

        
    feature_data.append(bursts_data)
    feature_data.append(bursts_length)
    feature_data.append(bursts_time)
    # feature_data.append(bursts_direction)
    feature_data.append(bursts_syn)
    feature_data.append(bursts_ack)
    feature_data.append(bursts_fin)
    feature_data.append(bursts_reset)
    feature_data.append(bursts_push)
    feature_data.append(bursts_urg)

    # 随机抽取samples_num个bursts
    if len(bursts_data) > samples_num:
        bursts_index = random.sample(range(len(bursts_data)), samples_num)
        feature_data = [[item[index] for index in bursts_index] for item in feature_data]

    return feature_data
        

def get_feature_packet(label_pcap,payload_len):
    feature_data = []

    packets = scapy.rdpcap(label_pcap)
    packet_data_string = ''  

    for packet in packets:
            packet_data = packet.copy()
            data = (binascii.hexlify(bytes(packet_data)))
            
            packet_string = data.decode()
            
            new_packet_string = packet_string[76:]
            
            packet_data_string += bigram_generation(new_packet_string, packet_len=payload_len, flag = True)

    feature_data.append(packet_data_string)
    return feature_data

def get_feature_flow(label_pcap, payload_len, payload_pac):
    
    feature_data = []
    packets = scapy.rdpcap(label_pcap)
    packet_count = 0  
    flow_data_string = '' 

    feature_result = extract(label_pcap, filter='tcp', extension=['tls.record.content_type', 'tls.record.opaque_type', 'tls.handshake.type'])
    if len(feature_result) == 0:
        feature_result = extract(label_pcap, filter='udp')
        if len(feature_result) == 0:
            return -1
        extract_keys = list(feature_result.keys())[0]
        if len(feature_result[label_pcap, extract_keys[1], extract_keys[2]].ip_lengths) < 3:
            print("preprocess flow %s but this flow has less than 3 packets." % label_pcap)
            return -1
    elif len(packets) < 3:
        print("preprocess flow %s but this flow has less than 3 packets." % label_pcap)
        return -1
    try:
        if len(feature_result[label_pcap, 'tcp', '0'].ip_lengths) < 3:
            print("preprocess flow %s but this flow has less than 3 packets." % label_pcap)
            return -1
    except Exception as e:
        print("*** this flow begings from 1 or other numbers than 0.")
        for key in feature_result.keys():
            if len(feature_result[key].ip_lengths) < 3:
                print("preprocess flow %s but this flow has less than 3 packets." % label_pcap)
                return -1

    if feature_result.keys() == {}.keys():
        return -1

    packet_length = []
    packet_time = []
    packet_direction = []
    packet_message_type = []
    
    if feature_result == {}:
        return -1
    feature_result_lens = len(feature_result.keys())
    for key in feature_result.keys():
        value = feature_result[key]
        packet_length.append(value.ip_lengths)
        packet_time.append(value.ip_timestamps)

        if len(packet_length) < feature_result_lens:
            continue
        elif len(packet_length) == 1:
            pass
        else:
            packet_length = [sum(packet_length, [])]
            packet_time = [sum(packet_time, [])]

        extension_dict = {}
        
        for len_index in range(len(packet_length)):
            extension_list = [0]*(len(packet_length[len_index]))

        extensions = value.extension
        
        if 'tls.record.content_type' in extensions.keys():
            for record_content in extensions['tls.record.content_type']:
                packet_index = record_content[1]
                ms_type = []
                
                if len(record_content[0]) > 2:
                    ms_type.extend(record_content[0].split(','))
                else:
                    ms_type.append(record_content[0])
                
                extension_dict[packet_index] = ms_type
            
            if 'tls.handshake.type' in extensions.keys():
                for tls_handshake in extensions['tls.handshake.type']:
                    packet_index = tls_handshake[1]
                    if packet_index not in extension_dict.keys():
                        continue
                    ms_type = []
                    if len(tls_handshake[0]) > 2:
                        ms_type.extend(tls_handshake[0].split(','))
                    else:
                        ms_type.append(tls_handshake[0])
                    source_length = len(extension_dict[packet_index])
                    for record_index in range(source_length):
                        if extension_dict[packet_index][record_index] == '22':
                            for handshake_type_index in range(len(ms_type)):
                                extension_dict[packet_index][record_index] = '22:' + ms_type[handshake_type_index]
                                if handshake_type_index > 0:
                                    extension_dict[packet_index].insert(handshake_type_index,
                                                                        ('22:' + ms_type[handshake_type_index]))
                            break
        if 'tls.record.opaque_type' in extensions.keys():
            for record_opaque in extensions['tls.record.opaque_type']:
                packet_index = record_opaque[1]
                ms_type = []
                if len(record_opaque[0]) > 2:
                    ms_type.extend(record_opaque[0].split(","))
                else:
                    ms_type.append(record_opaque[0])
                if packet_index not in extension_dict.keys():
                    extension_dict[packet_index] = ms_type
                else:
                    extension_dict[packet_index].extend(ms_type)

        extension_string_dict = {}
        for key in extension_dict.keys():
            temp_string = ''
            for status in extension_dict[key]:
                temp_string += status+','
            temp_string = temp_string[:-1]
            extension_string_dict[key] = temp_string
        
        is_source = 0
        if is_source:
            packet_message_type.append(extension_string_dict)
        else:
            for key in extension_dict.keys():
                if len(set(extension_dict[key])) == 1 and len(extension_dict[key]) > 1:
                    try:
                        extension_list[key] += len(extension_dict[key])
                    except Exception as e:
                        print(key)
                else:
                    for status in extension_dict[key]:
                        if ':' in status:
                            
                            extension_list[key - 1] += reduce(operator.mul, [int(x) for x in status.split(':')], 1)
                        else:
                           
                            if key <= len(packet_length[0]):
                                extension_list[key - 1] += int(status)
                            else:
                                with open("error_while_writin_record","a") as f:
                                    f.write(label_pcap + '\n')
                                continue
            packet_message_type.append(extension_list)
    for length in packet_length[0]:
        if length > 0:
            packet_direction.append(1)
        else:
            packet_direction.append(-1)

    packet_index = 0
    for packet in packets:
        packet_count += 1
        if packet_count == payload_pac:
            packet_data = packet.copy()
            data = (binascii.hexlify(bytes(packet_data)))
            packet_string = data.decode()[76:]
            flow_data_string += bigram_generation(packet_string, packet_len=payload_len, flag = True)
            break
        else:
            packet_data = packet.copy()
            data = (binascii.hexlify(bytes(packet_data)))
            packet_string = data.decode()[76:]
            flow_data_string += bigram_generation(packet_string, packet_len=payload_len, flag = True)
    feature_data.append(flow_data_string)
    feature_data.append(packet_length[0])
    feature_data.append(packet_time[0])
    feature_data.append(packet_direction)
    feature_data.append(packet_message_type[0])

    return feature_data

def generation(pcap_path, samples, features, splitcap = False, payload_length = 128, payload_packet = 5, dataset_save_path = "I:\\ex_results\\", dataset_level = "flow"):
    if os.path.exists(dataset_save_path + "dataset.json"):
        print("the pcap file of %s is finished generating."%pcap_path)
        
        clean_dataset = 0
        
        re_write = 0

        if clean_dataset:
            with open(dataset_save_path + "\\dataset.json", "r") as f:
                new_dataset = json.load(f)
            pop_keys = ['1','10','16','23','25','71']
            print("delete domains.")
            for p_k in pop_keys:
                print(new_dataset.pop(p_k))
            
            change_keys = [str(x) for x in range(113, 119)]
            relation_dict = {}
            for c_k_index in range(len(change_keys)):
                relation_dict[change_keys[c_k_index]] = pop_keys[c_k_index]
                new_dataset[pop_keys[c_k_index]] = new_dataset.pop(change_keys[c_k_index])
            with open(dataset_save_path + "\\dataset.json", "w") as f:
                json.dump(new_dataset, fp=f, ensure_ascii=False, indent=4)
        elif re_write:
            with open(dataset_save_path + "\\dataset.json", "r") as f:
                old_dataset = json.load(f)
            os.renames(dataset_save_path + "\\dataset.json", dataset_save_path + "\\old_dataset.json")
            with open(dataset_save_path + "\\new-samples.txt", "r") as f:
                source_samples = f.read().split('\n')
            new_dataset = {}
            samples_count = 0
            for i in range(len(source_samples)):
                current_class = source_samples[i].split('\t')
                if int(current_class[1]) > 9:
                    new_dataset[str(samples_count)] = old_dataset[str(i)]
                    samples_count += 1
                    print(old_dataset[str(i)]['samples'])
            with open(dataset_save_path + "\\dataset.json", "w") as f:
                json.dump(new_dataset, fp=f, ensure_ascii=False, indent=4)
        X, Y = obtain_data(pcap_path, samples, features, dataset_save_path)
        return X,Y

    dataset = {}
    
    label_name_list = []

    session_pcap_path  = {}

    for parent, dirs, files in os.walk(pcap_path):
        # 所有子目录的名称，即所有类流量名称
        if label_name_list == []:
            label_name_list.extend(dirs)

        tls13 = 0
        if tls13:
            record_file = dataset_save_path + "picked_file_record"
            target_path = dataset_save_path + "packet_splitcap\\"
            # 检查packet_splitcap目录下是否有文件
            if not os.path.getsize(target_path):
                with open(record_file, 'r') as f:
                    # 一次性读取整个文件，并以换行符分割。每一行表示一个文件路径
                    record_files = f.read().split('\n')
                for file in record_files[:-2]:
                    current_path = target_path + file.split('\\')[5]
                    new_name = '_'.join(file.split('\\')[6:])
                    if not os.path.exists(current_path):
                        os.mkdir(current_path)
                    # 将文件复制到packet_splitcap目录下
                    shutil.copyfile(file, os.path.join(current_path, new_name))

        for dir in label_name_list:
            for p,dd,ff in os.walk(parent + "\\" + dir):
                
                if splitcap:
                    for file in ff:
                        session_path = (split_cap(pcap_path, p + "\\" + file, file.split(".")[-2], dir, dataset_level = dataset_level))
                    session_pcap_path[dir] = pcap_path + "\\splitcap\\" + dir
                else:
                    # 每个类别目录下还有file.split(".")[-2]的目录，其中放的是会话的pcap文件
                    session_pcap_path[dir] = pcap_path + dir
        break

    # 将文本标签转化为数字标签
    label_id = {}
    for index in range(len(label_name_list)):
        label_id[label_name_list[index]] = index

    r_file_record = []
    print("\nBegin to generate features.")

    # 计数器，从第0个类别开始
    label_count = 0
    # 遍历每个类别
    for key in tqdm.tqdm(session_pcap_path.keys()):

        if dataset_level == "flow":
            if splitcap:
                for p, d, f in os.walk(session_pcap_path[key]):
                    # 一种流量中的多个流量
                    for file in f:
                        file_size = float(size_format(os.path.getsize(p + "\\" + file)))
                        # 2KB
                        if file_size < 5:
                            os.remove(p + "\\" + file)
                            print("remove sample: %s for its size is less than 5 KB." % (p + "\\" + file))

            # 预先在一个空的dataset中为每个类别生成一个特征字典
            if label_id[key] not in dataset:
                dataset[label_id[key]] = {
                    "samples": 0,
                    "payload": {},
                    "length": {},
                    "time": {},
                    "direction": {},
                    "message_type": {}
                }
        elif dataset_level == "burst":
            if splitcap:
                for p, d, f in os.walk(session_pcap_path[key]):
                    for file in f:
                        file_size = float(size_format(os.path.getsize(p + "\\" + file)))
                        # 2KB
                        if file_size < 5:
                            os.remove(p + "\\" + file)
                            print("remove sample: %s for its size is less than 5 KB." % (p + "\\" + file))

            # 预先在一个空的dataset中为每个类别生成一个特征字典
            if label_id[key] not in dataset:
                dataset[label_id[key]] = {
                    "samples": 0,
                    "payload": {},
                    "length": {},
                    "time": {},
                    # "direction": {},
                    "syn": {},
                    "ack": {},
                    "fin": {},
                    "rst": {},
                    "psh": {},
                    "urg": {}
                }
        elif dataset_level == "packet":
            if splitcap:# not splitcap
                for p, d, f in os.walk(session_pcap_path[key]):
                    for file in f:
                        current_file = p + "\\" + file
                        # 过滤小流量，里面没有有效信息
                        if not os.path.getsize(current_file):
                            os.remove(current_file)
                            print("current pcap %s is 0KB and delete"%current_file)
                        else:
                            current_packet = scapy.rdpcap(p + "\\" + file)
                            file_size = float(size_format(os.path.getsize(p + "\\" + file)))
                            try:
                                if 'TCP' in str(current_packet.res):
                                    # 0.12KB
                                    if file_size < 0.14:
                                        os.remove(p + "\\" + file)
                                        print("remove TCP sample: %s for its size is less than 0.14KB." % (
                                                    p + "\\" + file))
                                elif 'UDP' in str(current_packet.res):
                                    if file_size < 0.1:
                                        os.remove(p + "\\" + file)
                                        print("remove UDP sample: %s for its size is less than 0.1KB." % (
                                                    p + "\\" + file))
                            except Exception as e:
                                print("error in data_generation 611: scapy read pcap and analyse error")
                                os.remove(p + "\\" + file)
                                print("remove packet sample: %s for reading error." % (p + "\\" + file))
            if label_id[key] not in dataset:
                dataset[label_id[key]] = {
                    "samples": 0,
                    "payload": {}
                }
        if splitcap:
            continue

        target_all_files = [x[0] + "\\" + y for x in [(p, f) for p, d, f in os.walk(session_pcap_path[key])] for y in x[1]]
        # 随机选取samples个文件
        r_files = random.sample(target_all_files, samples[label_count])
        label_count += 1
        # 提取包或流的特征
        for r_f in r_files:
            if dataset_level == "flow":
                feature_data = get_feature_flow(r_f, payload_len=payload_length, payload_pac=payload_packet)
            elif dataset_level == "packet":
                feature_data = get_feature_packet(r_f, payload_len=payload_length)
            elif dataset_level == "burst":
                feature_data = get_feature_bursts(r_f, payload_len=payload_length, payload_pac=payload_packet, samples_num=max(samples))

            if feature_data == -1:
                continue
            r_file_record.append(r_f)

            # burst需要从flow中分割出多个包，所以这里不用更新samples
            if dataset_level != "burst":
                dataset[label_id[key]]["samples"] += 1
            if len(dataset[label_id[key]]["payload"].keys()) > 0:
                # payload，bigram格式，流为所有包的tcp payload拼接
                if dataset_level == "packet":
                    dataset[label_id[key]]["payload"][str(dataset[label_id[key]]["samples"])] = \
                        feature_data[0]
                elif dataset_level == "flow":
                    dataset[label_id[key]]["length"][str(dataset[label_id[key]]["samples"])] = \
                        feature_data[1]
                    dataset[label_id[key]]["time"][str(dataset[label_id[key]]["samples"])] = \
                        feature_data[2]
                    dataset[label_id[key]]["direction"][str(dataset[label_id[key]]["samples"])] = \
                        feature_data[3]
                    dataset[label_id[key]]["message_type"][str(dataset[label_id[key]]["samples"])] = \
                        feature_data[4]
                elif dataset_level == "burst":
                    for burst_index in range(len(feature_data[0])):
                        dataset[label_id[key]]["samples"] += 1
                        dataset[label_id[key]]["payload"][str(dataset[label_id[key]]["samples"])] = \
                            feature_data[0][burst_index]
                        dataset[label_id[key]]["length"][str(dataset[label_id[key]]["samples"])] = \
                            feature_data[1][burst_index]
                        dataset[label_id[key]]["time"][str(dataset[label_id[key]]["samples"])] = \
                            feature_data[2][burst_index]
                        dataset[label_id[key]]["syn"][str(dataset[label_id[key]]["samples"])] = \
                            feature_data[3][burst_index]
                        dataset[label_id[key]]["ack"][str(dataset[label_id[key]]["samples"])] = \
                            feature_data[4][burst_index]
                        dataset[label_id[key]]["fin"][str(dataset[label_id[key]]["samples"])] = \
                            feature_data[5][burst_index]
                        dataset[label_id[key]]["rst"][str(dataset[label_id[key]]["samples"])] = \
                            feature_data[6][burst_index]
                        dataset[label_id[key]]["psh"][str(dataset[label_id[key]]["samples"])] = \
                            feature_data[7][burst_index]
                        dataset[label_id[key]]["urg"][str(dataset[label_id[key]]["samples"])] = \
                            feature_data[8][burst_index]
            else:
                # 类别的第一个样本的特征
                dataset[label_id[key]]["payload"]["1"] = feature_data[0]
                if dataset_level == "flow":
                    dataset[label_id[key]]["length"]["1"] = feature_data[1]
                    dataset[label_id[key]]["time"]["1"] = feature_data[2]
                    dataset[label_id[key]]["direction"]["1"] = feature_data[3]
                    dataset[label_id[key]]["message_type"]["1"] = feature_data[4]

    # 所有类别的总样本数
    all_data_number = 0
    for index in range(len(label_name_list)):
        print("%s\t%s\t%d"%(label_id[label_name_list[index]], label_name_list[index], dataset[label_id[label_name_list[index]]]["samples"]))
        all_data_number += dataset[label_id[label_name_list[index]]]["samples"]
    print("all\t%d"%(all_data_number))

    with open(dataset_save_path + "\\picked_file_record","w") as p_f:
        # 写入文件名路径
        for i in r_file_record:
            p_f.write(i)
            p_f.write("\n")
    with open(dataset_save_path + "\\dataset.json", "w") as f:
        json.dump(dataset,fp=f,ensure_ascii=False,indent=4)

    X,Y = obtain_data(pcap_path, samples, features, dataset_save_path, json_data = dataset)
    return X,Y

def read_data_from_json(json_data, features, samples):
    X,Y = [], []
    ablation_flag = 0
    # 遍历所有特征
    for feature_index in range(len(features)):
        x = []
        label_count = 0
        for label in json_data.keys():
            sample_num = json_data[label]["samples"]
            if X == []:
                if not ablation_flag:
                    y = [label] * sample_num
                    Y.append(y)
                else:
                    if sample_num > 1500:
                        y = [label] * 1500
                    else:
                        y = [label] * sample_num
                    Y.append(y)
            if samples[label_count] < sample_num:
                x_label = []
                for sample_index in random.sample(list(json_data[label][features[feature_index]].keys()),1500):
                    x_label.append(json_data[label][features[feature_index]][sample_index])
                x.append(x_label)
            else:
                x_label = []
                for sample_index in json_data[label][features[feature_index]].keys():
                    # x_label的每一个元素对应一个样本的一个特征
                    x_label.append(json_data[label][features[feature_index]][sample_index])
                # x的每一个元素对应一个特征
                x.append(x_label)
            label_count += 1
        # 3维，【特征，标签，样本】
        X.append(x)
    return X,Y

def obtain_data(pcap_path, samples, features, dataset_save_path, json_data = None):
    
    if json_data:
        X,Y = read_data_from_json(json_data,features,samples)
    else:
        print("read dataset from json file.")
        with open(dataset_save_path + "\\dataset.json","r") as f:
            dataset = json.load(f)
        X,Y = read_data_from_json(dataset,features,samples)

    for index in range(len(X)):
        if len(X[index]) != len(Y):
            print("data and labels are not properly associated.")
            print("x:%s\ty:%s"%(len(X[index]),len(Y)))
            return -1
    return X,Y

def combine_dataset_json():
    dataset_name = "I:\\traffic_pcap\\splitcap\\dataset-"
    # dataset vocab
    dataset = {}
    # progress
    progress_num = 8
    for i in range(progress_num):
        dataset_file = dataset_name + str(i) + ".json"
        with open(dataset_file,"r") as f:
            json_data = json.load(f)
        for key in json_data.keys():
            if i > 1:
                new_key = int(key) + 9*1 + 6*(i-1)
            else:
                new_key = int(key) + 9*i
            print(new_key)
            if new_key not in dataset.keys():
                dataset[new_key] = json_data[key]
    with open("I:\\traffic_pcap\\splitcap\\dataset.json","w") as f:
        json.dump(dataset, fp=f, ensure_ascii=False, indent=4)
    return 0

def pretrain_dataset_generation(pcap_path):
    output_split_path = "I:\\dataset\\"
    pcap_output_path = "I:\\dataset\\"
    
    if not os.listdir(pcap_output_path):
        print("Begin to convert pcapng to pcap.")
        for _parent,_dirs,files in os.walk(pcap_path):
            for file in files:
                if 'pcapng' in file:
                    #print(_parent + file)
                    convert_pcapng_2_pcap(_parent, file, pcap_output_path)
                else:
                    shutil.copy(_parent+"\\"+file, pcap_output_path+file)
    
    if not os.path.exists(output_split_path + "splitcap"):
        print("Begin to split pcap as session flows.")
        
        for _p,_d,files in os.walk(pcap_output_path):
            for file in files:
                split_cap(output_split_path,_p+file,file)
    print("Begin to generate burst dataset.")
    # burst sample
    for _p,_d,files in os.walk(output_split_path + "splitcap"):
        for file in files:
            get_burst_feature(_p+"\\"+file, payload_len=64)
    return 0

def size_format(size):
    # 'KB'
    file_size = '%.3f' % float(size/1000)
    return file_size

if __name__ == '__main__':
    # pretrain
    pcap_path = "I:\\pcaps\\"
    # tls 13 downstream
    #pcap_path, samples, features = "I:\\dataset\\labeled\\", 500, ["payload","length","time","direction","message_type"]
    #X,Y = generation(pcap_path, samples, features, splitcap=False)
    # pretrain data
    pretrain_dataset_generation(pcap_path)
    #print("X:%s\tx:%s\tY:%s"%(len(X),len(X[0]),len(Y)))
    # combine dataset.json
    #combine_dataset_json()

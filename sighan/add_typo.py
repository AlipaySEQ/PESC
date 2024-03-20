import json
import string
import copy
import random
import numpy as np
import re
import math
from pypinyin import pinyin, lazy_pinyin
from pypinyin_dict.phrase_pinyin_data import di
from tqdm import tqdm
random.seed(1)
np.random.seed(1)

def is_Chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False


def add_typo(data_path, out_path, confusion):
    confusion_list = {}
    c_f = open(confusion, encoding="utf8")
    f = open(data_path, encoding="utf8")
    o_f = open(out_path, "w", encoding="utf8")
    lines = c_f.read().split('\n')
    d_lines = f.read().split('\n')[:-1]
    typo_lines = copy.deepcopy(d_lines)
    edit = [[] for _ in range(len(d_lines))]
    detect = [[] for _ in range(len(d_lines))]
    first_pinyin = [[] for _ in range(len(d_lines))]
    for line in lines:
        pairs = line.split(":")
        confusion_list[pairs[0]] = pairs[1]
    for line_index, line in tqdm(enumerate(d_lines)):
        pairs = line.split("/")
        
      
        detect[line_index] = eval(pairs[4])
        first_pinyin[line_index] = eval(pairs[5])
    
        typo_lines[line_index] = typo_lines[line_index].split("/")[0]
        for i, char in enumerate(pairs[0]):
            flag = random.random()
            if is_Chinese(char) == False or len(lazy_pinyin(char)[0]) == 1 or flag > 0.04:
                continue
            else:
                typo_char = random.choice(str(confusion_list.get(char)))
                ch_list = list(typo_lines[line_index])
                ch_list[i] = typo_char
                typo_lines[line_index] = ''.join(ch_list)
                edit[line_index].append((i, char))
                

    for t_line, (idx, line) in zip(typo_lines, enumerate(d_lines)):
        pairs = line.split("/")
        o_f.writelines(t_line + "/" + pairs[1] + "/" + str(edit[idx]) + "/" + pairs[2] + "/" + pairs[3]+"/"+str(detect[idx])+"/"+str(first_pinyin[idx]))
        o_f.writelines("\n")


def gen_pinyin_typo(data_path, out_path, typo_rate):
    # 生成拼音错误
    lower = string.ascii_lowercase
    letters = list(lower)
    count = 0
    py_candidate = string.ascii_lowercase
    n_f = open(out_path, "w", encoding='utf8')
    f = open(data_path, encoding='utf8')
    lines = f.read().strip().split('\n')
    edit = [[] for _ in range(len(lines))]
    char_error_list = [[] for _ in range(len(lines))]
    new_line = []
    for line in lines:
        pairs  = line.split('/')
        new_line.append(pairs[0])
    typo_lines = copy.deepcopy(new_line)
    edited = [0] * len(lines)
    visited = [0] * len(lines)
    segment_index = [[] for _ in range(len(lines))]
    detect = [[] for _ in range(len(lines))]
    #first_pinyin = [[] for _ in range(len(lines))]
    for line_index, line in tqdm(enumerate(lines)):
        used = [0] * len(line)
        move = 0
        pairs  = line.split('/')
        char_id= eval(pairs[3])
        char_error = eval(pairs[2])
        for i, char in enumerate(pairs[0]):
            # if used[i]:
            #     continue
            # if char.encode('UTF-8').isalpha():
            #     segment_index[line_index].append(0)
            #     used[i] = 1
            #     j = i + 1
            #     while j < len(line) and line[j].encode('UTF-8').isalpha():
            #         used[j] = 1
            #         if j == len(line)-1:
            #             segment_index[line_index].append(1)
            #             break
            #         if not line[j + 1].encode('UTF-8').isalpha():
            #             segment_index[line_index].append(1)
            #         else:
            #             segment_index[line_index].append(0)
            #         j += 1


            # else:
            segment_index[line_index].append(1)

            
            
            if i in char_id:
                for item in char_error:
                    if item[0] == i:
                        char_error_list[line_index].append((item[0]+move,item[1]))
                continue
            flag = random.random()
            if is_Chinese(char) == False or len(lazy_pinyin(char)[0]) == 1 or flag > typo_rate :
                detect[line_index].append(0)
                
                
            else:
                py_index = i
                typo = list(lazy_pinyin(char)[0])
                fi = letters.index(typo[0])+2
                if len(typo) == 1:
                    count += 1
                else:
                    index = random.choice(range(1, len(typo)))
                    error_type = np.random.choice(range(6), p=[0.3334,  0.3334, 0.134, 0.0664, 0.0664,
                                                               0.0664])  # 0:正确拼音 1：缩写 2:插入错误  3:删除错误  4:替换错误  5:换位错误
                    # error_type = np.random.choice(range(6), p=[0,  0, 0, 0,0,
                    #                                            1])  # 0:正确拼音 1：缩写 2:插入错误  3:删除错误  4:替换错误  5:换位错误
                    if error_type == 0:
                        pass
                    elif error_type == 1:
                        typo = typo[0:1]
                    elif error_type == 2:
                        typo.insert(index, random.choice(py_candidate))
                    elif error_type == 3:
                        typo.pop(index)
                    elif error_type == 4:
                        typo[index] = random.choice([i for i in py_candidate if i != typo[index]])
                    else:
                        typo[index], typo[index - 1] = typo[index - 1], typo[index]
                    error_char = ''.join(typo)


                    for a in range(len(error_char) - 1):
                        segment_index[line_index].append(0)
                    # for a in range(len(error_char)):
                    #     detect[line_index].insert(i + move, 1)
                    for a in range(len(error_char)):
                        if a == len(error_char) - 1:
                            detect[line_index].insert(i + move, fi)
                        else:
                            detect[line_index].insert(i + move, 1)
                    # for a in range(len(error_char)):
                    #     if a == len(error_char) - 1:
                    #         first_pinyin[line_index].insert(i + move, 1)
                    #     else:
                    #         first_pinyin[line_index].insert(i + move, 0)
                    move += len(error_char) - 1
                    if visited[line_index] == 1:
                        py_index += edited[line_index]
                    visited[line_index] = 1
                    ch_list = list(typo_lines[line_index])
                    char = ch_list[py_index]
                    ch_list[py_index] = error_char
                    edited[line_index] += len(error_char) - 1
                    new_ch_seq = ''.join(ch_list)
                    if len(error_char) > 1:
                        edit[line_index].append(((py_index, py_index + len(error_char) - 1), char))
                    else:
                        edit[line_index].append((py_index, char))
                    typo_lines[line_index] = new_ch_seq
                   
                    
       
    for (index, line), typo_line in zip(enumerate(lines), typo_lines):
        pairs  = line.split('/')
       
        n_f.writelines(typo_line + '/' + pairs[1] + "/" + str(char_error_list[index]) + "/"+ str(edit[index]) + "/" + str(segment_index[index]) +"/"+str(detect[index])+"\n")


# gen_pinyin_typo("dev.txt", "new_dev.txt", 0.06)
# add_typo("new_dev.txt", "./bert/data/dev.csv", "confusion.txt")
#gen_pinyin_typo("sighan15_test_.txt", "sighan15_test.csv", 0.06)
gen_pinyin_typo("train_.txt", "train.csv", 0.06)
gen_pinyin_typo("test_.txt", "test.csv", 0.06)
#add_typo("new_test.txt", "./bert/data/new_test.csv", "confusion.txt")
# gen_pinyin_typo("train.txt", "new_train.txt", 0.06)
# add_typo("new_train.txt", "./bert/data/train.csv", "confusion.txt")

# gen_pinyin_typo("test.txt", "new_test.txt", 0.06)
# add_typo("new_test.txt", "/opt/data/private/phvec/new_data/2.txt", "confusion.txt")


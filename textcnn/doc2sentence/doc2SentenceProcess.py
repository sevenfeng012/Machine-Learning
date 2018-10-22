#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-10-18 16:26:52
# @Author  : seven
# @Emial   : fengzhijian012@163.com
# @Link    : www.wortyby.com
# @Version : Version_1.0.0
# @Company : 阿基米德

import os
import jieba
import random


class doc2Sentence(object):

    def __init__(self, foldPath, limitDirFiles, dims=1000):
        '''[将文档生成没有停用词的句子]

        [foldPath 是训练样本存放的地方,通过遍历文件夹里的文档，将其进行分词存放。单个文件夹下面存放文件不能超过 limitDirFiles 个]

        Arguments:
                foldPath {[string]} -- [训练样本存放路径]
                limitDirFiles {[type]} -- [文件夹里面的文本限制个数]
                dims {[number]} -- [词向量的维数]
        '''
        super(doc2Sentence, self).__init__()
        self.foldPath = foldPath
        self.limitDirFiles = limitDirFiles
        self.dims = dims

    def distributeWordFrequency(self, data_list):
        all_words_dict = {}

        for word_list in data_list:
            for word in word_list:
                if word in all_words_dict.keys():
                    all_words_dict[word] += 1
                else:
                    all_words_dict[word] = 1

        all_words_tuple_list = sorted(
            all_words_dict.items(), key=lambda f: f[1], reverse=True)

        all_words_list, all_words_nums = zip(*all_words_tuple_list)

        all_words_list = list(all_words_list)

        return all_words_list

    def TextProcessing(self, foldPath, test_size=0.2):
        '''[处理生成向量文本]

        [假设foldpath 下面都是文件夹，子文件夹下面都是文本文件]

        Arguments:
                foldPath {[string]} -- [训练样本存放路径]

        Keyword Arguments:
                test_size {number} -- [训练集与测试集数据比例] (default: {0.2})
        '''
        folder_list = os.listdir(foldPath)
        data_list = []
        class_list = []

        for folder in folder_list:
            new_folder_path = os.path.join(foldPath, folder)
            files = os.listdir(new_folder_path)

            j = 1

            for file in files:
                if j > self.limitDirFiles:
                    break
                with open(os.path.join(new_folder_path, file), 'r', encoding='utf-8') as f:
                    raw = f.read()

                word_cut = jieba.cut(raw, cut_all=False)
                word_list = list(word_cut)

                data_list.append(word_list)
                class_list.append(folder)

                j += 1

        data_class_list = list(zip(data_list, class_list))  # 将数据与标签按照对应关系压缩合并
        random.shuffle(data_class_list)  # 打乱data_class_list的顺序
        index = int(len(data_class_list) * test_size) + 1  # 训练集与测试数据集切分的索引值
        train_list = data_class_list[index:]
        test_list = data_class_list[:index]

        train_data_list, train_class_list = zip(*train_list)
        test_data_list, test_class_list = zip(*test_list)

        all_words_list = self.distributeWordFrequency(train_data_list)

        return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list

    def MakeWordsSet(self, words_file):
        '''[文本去重]

        [将当前文本里的字按行度读取并去重]

        Arguments:
                words_file {[type]} -- [文本内容]

        Returns:
                [type] -- [去重后的文字集合]
        '''
        word_set = set()
        with open(words_file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                word = line.strip()

                if len(word) > 0:
                    word_set.add(word)
        return word_set

    def TextFeature(self, train_data_list, test_data_list, feature_words):
        def text_features(text, feature_words):
            text_words = set(text)
            features = [
                1 if word in text_words else 0 for word in feature_words]
            return features

        train_feature_list = [text_features(
            text, feature_words) for text in train_data_list]
        test_feature_list = [text_features(
            text, feature_words) for text in test_data_list]

        return train_feature_list, test_feature_list

    def words_dict(self, all_words_list, deleteN, stopwords_set=set()):
        feature_words = []
        n = 1

        for t in range(deleteN, len(all_words_list), 1):
            if n > self.dims:
                break
            if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(all_words_list[t]) < 5:
                feature_words.append(all_words_list[t])

            n += 1

        return feature_words

    def wordsToSentence(self, words, fileName):
        sentence = ''
        for word in words:
            sentence += word
        sentence += '\n'

        with open(fileName, 'a', encoding='utf-8') as f:
            f.write(sentence)

    def TransformerOldToNewDocOnline(self, folder, deleteN, fileName='feature.txt', test_size=0.2,Dir='.'):
        all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = self.TextProcessing(
            folder, test_size)

        stopwords_file = './stopwords_cn.txt'
        stopwords_set = self.MakeWordsSet(stopwords_file)

        class_list_index = 0

        for class_name in train_class_list:
            data_list_with_class = train_data_list[class_list_index]

            class_list_index += 1

            feature_words = data_list_with_class

            print(feature_words)

            fileP = class_name + '_' + fileName
            if not os.path.exists(Dir):
            	os.mkdir(Dir)

            fileP = os.path.join(Dir,fileP)

            self.wordsToSentence(feature_words, fileP)


doc2s = doc2Sentence(100, 128)

doc2s.TransformerOldToNewDocOnline('Train/Sample/', 0, 'feature.txt', 0.2,Dir='../classification/production/sample/')

import numpy as np
import re
from tensorflow.contrib import learn
import os


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    # string = re.sub(r"\'s", " \'s", string)
    # string = re.sub(r"\'ve", " \'ve", string)
    # string = re.sub(r"n\'t", " n\'t", string)
    # string = re.sub(r"\'re", " \'re", string)
    # string = re.sub(r"\'d", " \'d", string)
    # string = re.sub(r"\'ll", " \'ll", string)
    # string = re.sub(r",", " , ", string)
    # string = re.sub(r"!", " ! ", string)
    # string = re.sub(r"\(", " \( ", string)
    # string = re.sub(r"\)", " \) ", string)
    # string = re.sub(r"\?", " \? ", string)
    # string = re.sub(r"\s{2,}", " ", string)
    return string.strip('\n')


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(
        open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(
        open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # other_examples = list(
    #     open(other_data_file, "r", encoding='utf-8').readlines())
    # other_examples = [s.strip() for s in other_examples]
    # Split by words
    x_text = positive_examples + negative_examples 
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    # other_labels = [[0, 0, 1] for _ in other_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    # y = np.concatenate([positive_labels, negative_labels], 0)

    print(positive_examples)
    print(negative_examples)

    print(y)
    print(y.shape)
    return [x_text, y]


def load_data_and_labels_withDir(Dir):
    """
    Loads MR polarity data from files ,splits the data into words and generates labels.
    Return split sentences and labels.
    """

    # Load data from files
    x_text = []
    n = 0
    file_list = os.listdir(Dir)
    count = len(file_list)

    unitArray = np.eye(count)
    ylist=[]

    for file in file_list:
        classification_files = list(
            open(os.path.join(Dir,file), "r", encoding='utf-8').readlines())
        classification_files = [s.strip() for s in classification_files]
        x_text += classification_files
        x_text = [clean_str(sent) for sent in x_text]

        labels = [unitArray[n] for _ in classification_files]

        ylist.append(labels)

        n+=1

    y=np.concatenate(ylist,0)
    print(y)
    print(y.shape)

    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    print(data, batch_size, num_epochs)
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def chinese_tokenizer(documents):
    for document in documents:
        yield [i for i in document]


def preprocess(positive_data_file, negative_data_file):
    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    x_text, y = load_data_and_labels(
        positive_data_file, negative_data_file)

    # Build vocabulary
    max_document_length = max([len(x) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(
        100, tokenizer_fn=chinese_tokenizer)
    x = np.array(list(vocab_processor.fit_transform(x_text)))

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(0.03 * float(len(y)))
    x_train, x_dev = x_shuffled[
        :dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[
        :dev_sample_index], y_shuffled[dev_sample_index:]

    del x, y, x_shuffled, y_shuffled

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, vocab_processor, x_dev, y_dev


def preprocessWidthDir(Dir):
    # Data Preparation
    # =====================

    # Load Data
    print("Loading data ...")
    x_text, y = load_data_and_labels_withDir(Dir)

    # Build vocabulary
    max_document_length = max([len(x) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(
        100, tokenizer_fn=chinese_tokenizer)
    x = np.array(list(vocab_processor.fit_transform(x_text)))

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation

    dev_sample_index = -1 * int(0.3 * float(len(y))) #小训练样本
    x_train, x_dev = x_shuffled[
        :dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[
        :dev_sample_index], y_shuffled[dev_sample_index:]

    del x, y, x_shuffled, y_shuffled

    print("Vocabulary Size : {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev spilt : {:d}/{:d}".format(len(y_train), len(y_dev)))

    return x_train, y_train, vocab_processor, x_dev, y_dev

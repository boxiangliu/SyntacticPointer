from logging import root
import os.path
import numpy as np
from collections import defaultdict, OrderedDict

from lightning.io.common import DIGIT_RE
from lightning.io.alphabet import Alphabet
from lightning.io.logger import get_logger
from lightning.io.common import PAD_CHAR, PAD, PAD_POS, PAD_TYPE, PAD_ID_CHAR, PAD_ID_TAG, PAD_ID_WORD
from lightning.io.common import ROOT, END, ROOT_CHAR, ROOT_POS, ROOT_TYPE, END_CHAR, END_POS, END_TYPE
from neuronlp2.io.conllx_data import NUM_SYMBOLIC_TAGS

_START_VOCAB = [PAD, ROOT, END]
NUM_SYMBOLIC_TAGS = 3

def create_alphabets(alphabet_directory, train_path, data_paths=None, max_vocabulary_size=100000, embedd_dict=None,min_occurance=1, normalize_digits=True):
    def expand_vocab():
        vocab_set = set(vocab_list)
        for data_path in data_paths:
            with open(data_path, "r") as file:
                for line in file:
                    # details see https://conll.uvt.nl/
                    # Some example lines: 
                    # 0       1               2       3       4       5       6       7       8       9  
                    # 1       Bay             _       PROPN   NNP     _       3       nn      _       _
                    # 2       Financial       _       PROPN   NNP     _       3       nn      _       _
                    # 3       Corp.           _       PROPN   NNP     _       15      nsubj   _       _
                    line = line.strip()
                    if len(line) == 0:
                        continue 

                    tokens = line.split("\t")
                    for char in tokens[1]:
                        char_alphabet.add(char)

                    word = DIGIT_RE.sub("0", tokens[1]) if normalize_digits else tokens[1]
                    pos = tokens[4] # fine-grained part-of-speech tags
                    type = tokens[7] # dependency relation to the head

                    pos_alphabet.add(pos)
                    type_alphabet.add(type)

                    if word not in vocab_set and (word in embedd_dict or word.lower() in embedd_dict):
                        vocab_set.add(word)
                        vocab_list.append(word)

    logger = get_logger("Create Alphabets")
    word_alphabet = Alphabet("word", default_value=True, singleton=True)
    char_alphabet = Alphabet("character", default_value=True)
    pos_alphabet = Alphabet("pos")
    type_alphabet = Alphabet("type")

    if not os.path.isdir(alphabet_directory):
        logger.info("Creating Alphabets: {}".format(alphabet_directory))

        char_alphabet.add(PAD_CHAR)
        pos_alphabet.add(PAD_POS)
        type_alphabet.add(PAD_TYPE)

        char_alphabet.add(ROOT_CHAR)
        pos_alphabet.add(ROOT_POS)
        type_alphabet.add(ROOT_TYPE)

        vocab = defaultdict(int)
        with open(train_path, "r") as file:
            for line in file:
                line = line.strip()
                if len(line) == 0:
                    continue 

                tokens = line.split("\t")
                for char in tokens[1]:
                    char_alphabet.add(char)

                word = DIGIT_RE.sub("0", tokens[1]) if normalize_digits else tokens[1]
                vocab[word] += 1

                pos = tokens[4]
                pos_alphabet.add(pos)

                type = tokens[7]
                type_alphabet.add(type)

        # collect singletons
        singletons = set([word for word, count in vocab.items() if count <= min_occurance])

        # if a singleton is in pretrained embedding dict, set the count to min_occur + c 
        if embedd_dict is not None:
            assert isinstance(embedd_dict, OrderedDict)
            for word in vocab.keys():
                if word in embedd_dict or word.lower() in embedd_dict:
                    vocab[word] += min_occurance

        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        logger.info("Total vocabulary size: {}".format(len(vocab_list)))
        logger.info("Total singleton size: {}".format(len(singletons)))
        vocab_list = [word for word in vocab_list if word in _START_VOCAB or vocab[word] > min_occurance]
        logger.info("Total vocabulary size (w.o. rare word): {}".format(len(vocab_list)))

        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]

        if data_paths is not None and embedd_dict is not None:
            expand_vocab()

        for word in vocab_list:
            word_alphabet.add(word)
            if word in singletons:
                word_alphabet.add_singleton(word_alphabet.get_index(word))
        
        word_alphabet.save(alphabet_directory)
        char_alphabet.save(alphabet_directory)
        pos_alphabet.save(alphabet_directory)
        type_alphabet.save(alphabet_directory)
    else:
        word_alphabet.load(alphabet_directory)
        char_alphabet.load(alphabet_directory)
        pos_alphabet.load(alphabet_directory)
        type_alphabet.load(alphabet_directory)

    word_alphabet.close()
    char_alphabet.close()
    pos_alphabet.close()
    type_alphabet.close()

    logger.info("Word Alphabet Size (Singleton): %d (%d)" % (word_alphabet.size(), word_alphabet.singleton_size()))
    logger.info("Character Alphabet Size: %d" % char_alphabet.size())
    logger.info("POS Alphabet Size: %d" % pos_alphabet.size())
    logger.info("Type Alphabet Size: %d" % type_alphabet.size())
    return word_alphabet, char_alphabet, pos_alphabet, type_alphabet


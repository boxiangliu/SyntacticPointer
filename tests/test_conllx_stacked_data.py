from lightning.io.conllx_stacked_data import (
    create_alphabets,
    read_data,
    read_bucketed_data,
)
from lightning import utils
import pytest

word_embedding = "sskip"
word_path = "experiments/embs/sskip.eng.100.gz"

alphabet_path = "tests/alphabets"
train_path = "experiments/data/ptb-train.conllx"
dev_path = "experiments/data/ptb-dev.conllx"
test_path = "experiments/data/ptb-test.conllx"


# @pytest.fixture
def word_dict():
    word_dict, word_dim = utils.load_embedding_dict(word_embedding, word_path)
    return word_dict


# @pytest.fixture
def alphabets(word_dict):
    word_alphabet, char_alphabet, pos_alphabet, type_alphabet = create_alphabets(
        alphabet_path,
        train_path,
        data_paths=[dev_path, test_path],
        embedd_dict=word_dict,
        max_vocabulary_size=200000,
    )

    return word_alphabet, char_alphabet, pos_alphabet, type_alphabet


def test_read_data(alphabets):
    word_alphabet, char_alphabet, pos_alphabet, type_alphabet = alphabets

    data_tensor, data_size = read_data(
        train_path,
        word_alphabet,
        char_alphabet,
        pos_alphabet,
        type_alphabet,
    )
    # data_tensor is a dictionary with the following keys:
    # dict_keys(['WORD', 'CHAR', 'POS', 'HEAD', 'TYPE', 'MASK_ENC', 'SINGLE', 'LENGTH', 'STACK_HEAD', 'CHILD', 'STACK_TYPE', 'MASK_DEC'])
    assert data_tensor["WORD"].shape[1] == 250
    assert data_tensor["CHAR"].shape[2] == 24
    assert data_size == 1921


def test_read_bucketed_data(alphabets):
    word_alphabet, char_alphabet, pos_alphabet, type_alphabet = alphabets

    data_tensor, data_size = read_bucketed_data(
        train_path,
        word_alphabet,
        char_alphabet,
        pos_alphabet,
        type_alphabet,
    )
    # data_tensor is a dictionary with the following keys:
    # dict_keys(['WORD', 'CHAR', 'POS', 'HEAD', 'TYPE', 'MASK_ENC', 'SINGLE', 'LENGTH', 'STACK_HEAD', 'CHILD', 'STACK_TYPE', 'MASK_DEC'])

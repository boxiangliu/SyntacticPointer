from lightning.io.conllx_data import create_alphabets, read_bucketed_data, read_data
from lightning import utils

word_embedding = "sskip"
word_path = "embs/sskip.eng.100.gz"

alphabet_path = "./tests/alphabets"
train_path = "data/ptb-train.conllx"
dev_path = "data/ptb-dev.conllx"
test_path = "data/ptb-test.conllx"


def test_load_embedding_dict():
    word_dict, word_dim = utils.load_embedding_dict(word_embedding, word_path)


def test_create_alphabets():
    word_alphabet, char_alphabet, pos_alphabet, type_alphabet = create_alphabets(
        alphabet_path,
        train_path,
        data_paths=[dev_path, test_path],
        embedd_dict=word_dict,
        max_vocabulary_size=200000,
    )


def test_read_data():
    read_data(train_path)

from lightning.io.conllx_data import create_alphabets, read_bucketed_data, read_data
from lightning import utils

word_embedding = "sskip"
word_path = "experiments/embs/sskip.eng.100.gz"

alphabet_path = "tests/alphabets"
train_path = "experiments/data/ptb-train.conllx"
dev_path = "experiments/data/ptb-dev.conllx"
test_path = "experiments/data/ptb-test.conllx"


def test_create_alphabets():
    word_dict, word_dim = utils.load_embedding_dict(word_embedding, word_path)

    word_alphabet, char_alphabet, pos_alphabet, type_alphabet = create_alphabets(
        alphabet_path,
        train_path,
        data_paths=[dev_path, test_path],
        embedd_dict=word_dict,
        max_vocabulary_size=200000,
    )

    # The Alphabet class contains all possible word/character/part-of-speech tage/edge types
    for word in ["_PAD", "for", "to"]:
        assert word in word_alphabet.instances

    for char in ["_PAD_CHAR", "A", "a"]:
        assert char in char_alphabet.instances

    for pos in ["_PAD_POS", "DT", "NN"]:
        assert pos in pos_alphabet.instances

    for type in ["_<PAD>", "det", "punct"]:
        assert type in type_alphabet.instances


def test_read_data():
    read_data(train_path)

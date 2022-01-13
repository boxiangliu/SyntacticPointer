from lightning import utils


word_embedding = "sskip"
word_path = "experiments/embs/sskip.eng.100.gz"


def test_load_embedding_dict():
    word_dict, word_dim = utils.load_embedding_dict(word_embedding, word_path)  # slow
    assert word_dim == 100
    assert len(word_dict) == 207693

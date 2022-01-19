from lightning.data import PTBData
import pytest

train_path = "experiments/data/ptb-train.conllx"
dev_path = "experiments/data/ptb-dev.conllx"
test_path = "experiments/data/ptb-test.conllx"
alphabet_path = "experiments/alphabet/"
word_embedding = "sskip"
word_path = "experiments/embs/sskip.eng.100.gz"


@pytest.fixture
def ptb_data():
    ptb_data = PTBData(
        train_path, dev_path, test_path, alphabet_path, word_embedding, word_path
    )
    ptb_data.setup()
    return ptb_data


def test_ptb_data(ptb_data):
    train_dataloader = ptb_data.train_dataloader()
    train_iter = iter(train_dataloader)
    batch_ = train_iter.next()
    assert batch_["WORD"].shape[0] == 32

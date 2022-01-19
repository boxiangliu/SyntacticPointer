from lighting import Parsing
from lightning.data import PTBData

train_path = "experiments/data/ptb-train.conllx"
dev_path = "experiments/data/ptb-dev.conllx"
test_path = "experiments/data/ptb-test.conllx"
alphabet_path = "experiments/alphabet/"
word_embedding = "sskip"
word_path = "experiments/embs/sskip.eng.100.gz"

ptb_data = PTBData(
    train_path, dev_path, test_path, alphabet_path, word_embedding, word_path
)
ptb_data.setup()

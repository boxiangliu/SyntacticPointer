from lightning.lightning_module import Parsing
from lightning.data import PTBData
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

train_path = "experiments/data/ptb-train.conllx"
dev_path = "experiments/data/ptb-dev.conllx"
test_path = "experiments/data/ptb-test.conllx"
alphabet_path = "experiments/alphabet/"
word_embedding = "sskip"
word_path = "experiments/embs/sskip.eng.100.gz"
config = "lightning/configs/l2r.json"
model_path = "experiments/models/"


ptb_data = PTBData(
    train_path, dev_path, test_path, alphabet_path, word_embedding, word_path
)

model = Parsing(
    config, train_path, dev_path, test_path, model_path, word_path, word_embedding
)

wandb_logger = WandbLogger()
trainer = pl.Trainer(
    max_epochs=10, accelerator="gpu", strategy="dp", logger=wandb_logger, gpus=[0]
)
trainer.fit(model, ptb_data)

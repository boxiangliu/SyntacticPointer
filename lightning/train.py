from lightning.lightning import Parsing
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
train_loader = ptb_data.train_dataloader()
train_iter = iter(train_loader)

config = "lightning/configs/l2r.json"
model_path = "experiments/models/"

model = Parsing(
    config, train_path, dev_path, test_path, model_path, word_path, word_embedding
)
batch = train_iter.next()

words = batch["WORD"]
chars = batch["CHAR"]
postags = batch["POS"]
heads = batch["HEAD"]
masks_enc = batch["MASK_ENC"]
masks_dec = batch["MASK_DEC"]
stacked_heads = batch["STACK_HEAD"]
children = batch["CHILD"]
siblings = batch["SIBLING"]
stacked_types = batch["STACK_TYPE"]
nbatch = words.size(0)
nwords = masks_enc.sum() - nbatch

loss_arc, loss_type = model.network.loss(
    words,
    chars,
    postags,
    heads,
    stacked_heads,
    children,
    siblings,
    stacked_types,
    mask_e=masks_enc,
    mask_d=masks_dec,
)

exit(0)

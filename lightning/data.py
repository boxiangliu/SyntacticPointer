from audioop import reverse
import pytorch_lightning as pl

from lightning.io import conllx_data, conllx_stacked_data
from lightning import utils
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class DictDataSet(Dataset):
    def __init__(self, data: dict, size: int):
        self.data = data
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        return dict([(k, v[idx]) for k, v in self.data.items()])


class PTBData(pl.LightningDataModule):
    def __init__(
        self,
        train_path,
        dev_path,
        test_path,
        alphabet_path,
        word_embedding,
        word_path,
        char_embedding="random",
        char_path=None,
        batch_size=32,
    ):
        super().__init__()
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path

        self.alphabet_path = alphabet_path
        self.word_embedding = word_embedding
        self.word_path = word_path
        self.char_embedding = char_embedding
        self.char_path = char_path
        self.batch_size = batch_size

    def setup(self):
        word_dict, word_dim = utils.load_embedding_dict(
            self.word_embedding, self.word_path
        )

        (
            word_alphabet,
            char_alphabet,
            pos_alphabet,
            type_alphabet,
        ) = conllx_data.create_alphabets(
            self.alphabet_path,
            self.train_path,
            data_paths=[self.dev_path, self.test_path],
            embedd_dict=word_dict,
            max_vocabulary_size=200000,
        )

        self.data_train, self.train_size = conllx_stacked_data.read_data(
            self.train_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet
        )

        self.data_dev, self.dev_size = conllx_stacked_data.read_data(
            self.dev_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet
        )

        self.data_test, self.test_size = conllx_stacked_data.read_data(
            self.test_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet
        )

    def train_dataloader(self):
        data_train = DictDataSet(self.data_train, self.train_size)
        return DataLoader(data_train, batch_size=self.batch_size)

    def val_dataloader(self):
        data_dev = DictDataSet(self.data_dev, self.dev_size)
        return DataLoader(data_dev, batch_size=self.batch_size)

    def test_dataloader(self):
        data_test = DictDataSet(self.data_test, self.test_size)
        return DataLoader(data_test, batch_size=self.batch_size)

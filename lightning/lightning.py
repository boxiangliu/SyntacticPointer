import os
import json
import numpy as np

import torch
from torch.optim import Adam

import pytorch_lightning as pl
from lightning.models.models import L2RPtrNet
from lightning import utils
from lightning.io.logger import get_logger
from lightning.io import conllx_data


class Parsing(pl.LightningModule):
    def __init__(
        self,
        config,
        train_path,
        dev_path,
        test_path,
        model_path,
        word_path,
        word_embedding="sskip",
        char_path=None,
        char_embedding="random",
        punctuation=[".", "``", "''", ":", ","],
        optim="Adam",
        learning_rate=1e-3,
        lr_decay=0.999997,
        beta1=0.9,
        beta2=0.9,
        eps=1e-4,
        weight_decay=0.0,
        seed=1234,
        loss_type="token",
        unk_replace=0.5,
        freeze=None,
    ):
        super().__init__()
        pl.seed_everything(seed)

        logger = get_logger("Parsing")

        self.seed = seed
        self.optim = optim
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.loss_ty_token = loss_type == "token"
        self.unk_replace = unk_replace
        self.freeze = freeze
        self.punctuation = punctuation

        self.word_embedding = word_embedding
        self.word_path = word_path
        self.char_embedding = char_embedding
        self.char_path = char_path

        word_dict, word_dim = utils.load_embedding_dict(
            self.word_embedding, self.word_path
        )
        if char_embedding != "random":
            char_dict, char_dim = utils.load_embedding_dict(
                self.char_embedding, self.char_path
            )
        else:
            char_dict = None
            char_dim = None

        logger.info("Creating alphabets")
        alphabet_path = os.path.join(model_path, "alphabets")
        (
            word_alphabet,
            char_alphabet,
            pos_alphabet,
            type_alphabet,
        ) = conllx_data.create_alphabets(
            alphabet_path,
            train_path,
            data_paths=[dev_path, test_path],
            embedd_dict=word_dict,
            max_vocabulary_size=200000,
        )

        num_words = word_alphabet.size()
        num_chars = char_alphabet.size()
        num_pos = pos_alphabet.size()
        num_types = type_alphabet.size()

        logger.info("Word Alphabet Size: %d" % num_words)
        logger.info("Character Alphabet Size: %d" % num_chars)
        logger.info("POS Alphabet Size: %d" % num_pos)
        logger.info("Type Alphabet Size: %d" % num_types)

        result_path = os.path.join(model_path, "tmp")
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        punct_set = None
        if punctuation is not None:
            punct_set = set(punctuation)
            logger.info("punctuations(%d): %s" % (len(punct_set), " ".join(punct_set)))

        def construct_word_embedding_table():
            scale = np.sqrt(3.0 / word_dim)
            table = np.empty([word_alphabet.size(), word_dim], dtype=np.float32)
            table[conllx_data.UNK_ID, :] = (
                np.zeros([1, word_dim]).astype(np.float32)
                if freeze
                else np.random.uniform(-scale, scale, [1, word_dim]).astype(np.float32)
            )
            oov = 0
            for word, index in word_alphabet.items():
                if word in word_dict:
                    embedding = word_dict[word]
                elif word.lower() in word_dict:
                    embedding = word_dict[word.lower()]
                else:
                    embedding = (
                        np.zeros([1, word_dim]).astype(np.float32)
                        if freeze
                        else np.random.uniform(-scale, scale, [1, word_dim]).astype(
                            np.float32
                        )
                    )
                    oov += 1
                table[index, :] = embedding
            print("word OOV: %d" % oov)
            return torch.from_numpy(table)

        def construct_char_embedding_table():
            if char_dict is None:
                return None

            scale = np.sqrt(3.0 / char_dim)
            table = np.empty([num_chars, char_dim], dtype=np.float32)
            table[conllx_data.UNK_ID, :] = np.random.uniform(
                -scale, scale, [1, char_dim]
            ).astype(np.float32)
            oov = 0
            for (char, index) in char_alphabet.items():
                if char in char_dict:
                    embedding = char_dict[char]
                else:
                    embedding = np.random.uniform(-scale, scale, [1, char_dim]).astype(
                        np.float32
                    )
                    oov += 1
                table[index, :] = embedding
            print("character OOV: %d" % oov)
            return torch.from_numpy(table)

        word_table = construct_word_embedding_table()
        char_table = construct_char_embedding_table()
        logger.info("constructing network...")

        hyps = json.load(open(config, "r"))
        json.dump(hyps, open(os.path.join(model_path, "config.json"), "w"), indent=2)

        model_type = hyps["model"]
        assert model_type == "L2RPtr"
        word_dim = hyps["word_dim"]
        char_dim = hyps["char_dim"]
        use_pos = hyps["pos"]
        pos_dim = hyps["pos_dim"]
        mode = hyps["rnn_mode"]
        hidden_size = hyps["hidden_size"]
        arc_space = hyps["arc_space"]
        type_space = hyps["type_space"]
        p_in = hyps["p_in"]
        p_out = hyps["p_out"]
        p_rnn = hyps["p_rnn"]
        activation = hyps["activation"]
        prior_order = None

        if model_type == "L2RPtr":
            encoder_layers = hyps["encoder_layers"]
            decoder_layers = hyps["decoder_layers"]
            num_layers = (encoder_layers, decoder_layers)
            prior_order = hyps["prior_order"]
            grandPar = hyps["grandPar"]
            sibling = hyps["sibling"]
            self.network = L2RPtrNet(
                word_dim,
                num_words,
                char_dim,
                num_chars,
                pos_dim,
                num_pos,
                mode,
                hidden_size,
                encoder_layers,
                decoder_layers,
                num_types,
                arc_space,
                type_space,
                embedd_word=word_table,
                embedd_char=char_table,
                prior_order=prior_order,
                activation=activation,
                p_in=p_in,
                p_out=p_out,
                p_rnn=p_rnn,
                pos=use_pos,
                grandPar=grandPar,
                sibling=sibling,
            )

        model = "{}-{}".format(model_type, mode)
        logger.info(
            "Network: %s, num_layer=%s, hidden=%d, act=%s"
            % (model, num_layers, hidden_size, activation)
        )
        logger.info(
            "dropout(in, out, rnn): %s(%.2f, %.2f, %s)"
            % ("variational", p_in, p_out, p_rnn)
        )
        logger.info(
            "# of Parameters: %d"
            % (sum([param.numel() for param in self.network.parameters()]))
        )

    def forward(self):
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
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

        loss_arc, loss_type = self.network.loss(
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
        loss_arc = loss_arc.sum()  # sum over batch
        loss_type = loss_type.sum()
        loss_total = loss_arc + loss_type

        if self.loss_ty_token:
            loss = loss_total.div(nwords)
        else:
            loss = loss_total.div(nbatch)

        return loss

    def validation_step(self):
        raise NotImplementedError()

    def test_step(self):
        raise NotImplementedError()

    def predict_step(self):
        raise NotImplementedError()
        # TODO: transfer the parsing::L2RPtrNet.decode() to forward()

    def configure_optimizers(self):
        if self.optim == "Adam":
            return Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                betas=(self.beta1, self.beta2),
                eps=self.eps,
            )
        else:
            raise NotImplementedError

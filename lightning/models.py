from torch import nn
import torch
from lightning.modules import CharCNN
from enum import Enum

class PriorOrder(Enum):
    DEPTH = 0
    INSIDE_OUT = 1
    LEFT2RIGTH = 2

class L2RPtrNet(nn.Module):
    def __init__(
        self,
        word_dim,
        num_words,
        char_dim,
        num_chars,
        pos_dim,
        num_pos,
        rnn_mode,
        hidden_size,
        encoder_layers,
        decoder_layers,
        num_labels,
        arc_space,
        type_space,
        embedd_word=None,
        embedd_char=None,
        embedd_pos=None,
        p_in=0.33,
        p_out=0.33,
        p_rnn=(0.33, 0.33),
        pos=True,
        prior_order="inside_out",
        grandPar=False,
        sibling=False,
        activation="elu",
        remove_cycles=False,
    ):
        super().__init__()
        self.word_embed = nn.Embedding(
            num_words, word_dim, _weight=embedd_word, padding_idx=1
        )
        self.pos_embed = (
            nn.Embedding(num_pos, pos_dim, _weight=embedd_pos, padding_idx=1)
            if pos
            else None
        )
        self.char_embed = nn.Embedding(
            num_chars, char_dim, _weight=embedd_char, padding_idx=1
        )
        self.char_cnn = CharCNN(
            2, char_dim, char_dim, hidden_channels=char_dim * 4, activation=activation
        )

        self.dropout_in = nn.Dropout2d(p=p_in)
        self.dropout_out = nn.Dropout2d(p=p_out)
        self.num_labels = num_labels

        if prior_order in ["deep_first", "shallow_first"]:
            self.prior_order = PriorOrder.DEPTH
        elif prior_order == "inside_out":
            self.prior_order = PriorOrder.INSIDE_OUT
        elif prior_order == "left2right":
            self.prior_order = PriorOrder.LEFT2RIGTH
        else:
            raise ValueError("Unknown prior order: {}".format(prior_order))

        self.grandPar = grandPar
        self.sibling = sibling
        self.remove_cycles = remove_cycles

        if rnn_mode == "RNN":
            RNN_ENCODER = VarRNN
            RNN_DECODER = VarRNN
        elif rnn_mode == "LSTM":
            RNN_ENCODER = VarLSTM
            RNN_DECODER = VarLSTM
        elif rnn_mode == "FastLSTM":
            RNN_ENCODER = VarFastLSTM
            RNN_DECODER = VarFastLSTM
        elif rnn_mode == "GRU":
            RNN_ENCODER = VarGRU
            RNN_DECODER = VarGRU
        else:
            raise ValueError("Unknown RNN mode: {}".format(rnn_mode))

        dim_enc = word_dim + char_dim
        if pos:
            dim_enc += pos_dim

        self.encoder_layers = encoder_layers
        self.encoder = RNN_ENCODER(
            dim_enc,
            hidden_size,
            num_layers=encoder_layers,
            batch_first=True,
            bidirectional=True,
            dropout=p_rnn,
        )

        dim_dec = hidden_size // 2
        self.src_dense = nn.Linear(2 * hidden_size, dim_dec)
        self.decoder_layers = decoder_layers
        self.decoder = RNN_DECODER(
            dim_dec,
            hidden_size,
            num_layers=decoder_layers,
            batch_first=True,
            bidirectional=False,
            dropout=p_rnn,
        )

        self.hx_dense = nn.Linear(2 * hidden_size, hidden_size)

        self.arc_h = nn.Linear(hidden_size, arc_space)  # arc dense for decoder
        self.arc_c = nn.Linear(hidden_size * 2, arc_space)  # arc dense for encoder
        self.biaffine = BiAffine(arc_space, arc_space)

        self.type_h = nn.Linear(hidden_size, type_space)  # type dense for decoder
        self.type_c = nn.Linear(hidden_size * 2, type_space)  # type dense for encoder
        self.bilinear = BiLinear(type_space, type_space, self.num_labels)

        assert activation in ["elu", "tanh"]
        if activation == "elu":
            self.activation = nn.ELU(inplace=True)
        else:
            self.activation = nn.Tanh()

        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.reset_parameters(embedd_word, embedd_char, embedd_pos)

    def reset_parameters(self, embedd_word, embedd_char, embedd_pos):
        raise NotImplementedError()

    def _get_encoder_output(self, input_word, input_char, input_pos, mask=None):
        # [batch, length, word_dim]
        word = self.word_embed(input_word)

        # [batch, length, char_length, char_dim]
        char = self.char_cnn(self.char_embed(input_char))

        # apply dropout word on input
        word = self.dropout_in(word)
        char = self.dropout_in(char)

        # concatenate word and char, [batch, length, word_dim + char_filter]
        enc = torch.cat([word, char], dim=2)

        if self.pos_embed is not None:
            # [batch, length, pos_dim]
            pos = self.pos_embed(input_pos)
            # apply dropout on input
            pos = self.dropout_in(pos)
            enc = torch.cat([enc, pos], dim=2)

        # output from rnn [batch, length, hidden_size]
        output, hn = self.encoder(enc, mask)

        # apply dropout
        # [batch, length, hidden_size] -> [batch, hidden_size, length] -> [batch, length, hidden_size]
        output = self.dropout_out(output.transpose(1, 2)).transpose(1, 2)

        return output, hn

    def _get_decoder_output(
        self, output_enc, heads, heads_stack, siblings, hx, mask=None
    ):
        # get vector for heads [batch, length_decoder, input_dim]
        enc_dim = output_enc.size(2)
        batch, length_dec = heads_stack.size()
        # rearrange the order of input tokens according to heads_stack:
        src_encoding = output_enc.gather(
            dim=1, index=heads_stack.unsqueeze(2).expand(batch, length_dec, enc_dim)
        )

        # transform to decoder input
        # [batch, length_decoder, dec_dim]
        src_encoding = self.activation(self.src_dense(src_encoding))
        # output from rnn [batch, length, hidden_size]
        output, hn = self.decoder(src, src_encoding, mask, hx=hx)
        # apply dropout
        # [batch, length, hidden_size] -> [batch, hidden_size, length] -> [batch, length, hidden_size]
        output = self.dropout_out(output.transpose(1, 2)).transpose(1, 2)

        return output, hn

    def _transform_decoder_init_state(self):
        hn, cn = hn
        # hn dimension: [2 * num_layers, batch, hidden_size], 2 because of bidirectional
        _, batch, hidden_size = cn.size()
        # take the last layers (one in each direction)
        cn = torch.cat([cn[-1], cn[-2]], dim=1).unsqueeze(0)
        # take hx_dense to [1, batch, hidden_size]
        cn = self.hx_dense(cn)
        # [decoder_layers, batch, hidden_size]
        if self.decoder_layers > 1:
            cn = torch.cat(
                [cn, cn.new_zeros(self.decoder_layers - 1, batch, hidden_size)], dim=0
            )
        # hn is tanh(cn)
        hn = torch.tanh(cn)
        hn = (hn, cn)
        return hn

    def loss(
        self,
        input_word,
        input_char,
        input_pos,
        heads,
        stacked_heads,
        children,
        siblings,
        stacked_types,
        mask_e=None,
        mask_d=None,
    ):
        # output from encoder [batch, length_encoder, hidden_size]
        output_enc, hn = self._get_encoder_output(
            input_word, input_char, input_pos, mask=mask_e
        )

        # output size [batch, length_encoder, arc_space]
        arc_c = self.activation(seelf.arc_c(output_enc))
        # output size [batch, length_encoder, type_space]
        type_c = self.activation(self.type_c(output_enc))

        # transform hn to [decoder_layers, batch, hidden_size]
        hn = self._transform_decoder_init_state(hn)

        # output from decoder [batch, length_decoder, tag_space]
        output_dec, _ = self._get_decoder_output(
            output_enc, heads, stacked_heads, siblings, hn, mask=mask_d
        )

        # output size [batch, length_decoder, arc_space]
        arc_h = self.activation(self.arc_h(output_dec))

        # output size [batch, length_decoder, type_space]
        type_h = self.activation(self.type_h(output_dec))

        batch, max_len_d, type_space = type_h.size()

        # apply dropout
        # [batch, length_decoder, hidden] + [batch, length_encoder, hidden] -> [batch, length_decoder + length_encoder, hidden]
        arc = self.dropout_out(
            torch.cat([arc_h, arc_c], dim=1).transpose(1, 2)
        ).transpose(1, 2)
        arc_h = arc[:, :max_len_d]
        arc_c = arc[:, max_len_d:]

        type = self.dropout_out(
            torch.cat([type_h, type_c], dim=1).transpose(1, 2)
        ).transpose(1, 2)
        type_h = type[:, :max_len_d]
        type_c = type[:, max_len_d:]

        # [batch, length_decoder, length_encoder]
        out_arc = self.biaffine(arc_h, arc_c, mask_query=mask_d, mask_key=mask_e)

        # get vector for heads [batch, length_decoder, type_space]
        # This statement gets the type vector for the head (assuming head is known).
        # children is a misnomer; children means head.
        type_c = type_c.gather(
            dim=1, index=children.unsqueeze(2).expand(batch, max_len_d, type_space)
        )

        # compute output for type [batch, length_decoder, num_labels]
        out_type = self.bilinear(type_h, type_c)

        # mask invalid position to -inf for log_softmax
        if mask_e is not None:
            minus_mask_e = mask_e.eq(0).unsqueeze(1)
            minus_mask_d = mask_d.eq(0).unsqueeze(2)
            out_arc = out_arc.masked_fill(minus_mask_d * minus_mask_e, float("-inf"))

        # loss_arc shape [batch, length_decoder]
        loss_arc = self.criterion(out_arc.transpose(1, 2), children)
        loss_type = self.criterion(out_type.transpose(1, 2), stacked_types)

        if mask_d is not None:
            loss_arc = loss_arc * mask_d
            loss_type = loss_type * mask_d

        # loss_arc [batch, length_decoder]; loss_type [batch, length_decoder]
        return loss_arc.sum(dim=1), loss_type.sum(dim=1)

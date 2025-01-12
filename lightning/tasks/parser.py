__author__ = "max"

import re
import numpy as np


def is_uni_punctuation(word):
    match = re.match("^[^\w\s]+$]", word, flags=re.UNICODE)
    return match is not None


def is_punctuation(word, pos, punct_set=None):
    if punct_set is None:
        return is_uni_punctuation(word)
    else:
        return pos in punct_set


def eval(
    words,
    postags,
    heads_pred,
    types_pred,
    heads,
    types,
    word_alphabet,
    pos_alphabet,
    lengths,
    punct_set=None,
    symbolic_root=False,
    symbolic_end=False,
):
    batch_size, _ = words.shape
    ucorr = 0.0
    lcorr = 0.0
    total = 0.0
    ucomplete_match = 0.0
    lcomplete_match = 0.0

    ucorr_nopunc = 0.0
    lcorr_nopunc = 0.0
    total_nopunc = 0.0
    ucomplete_match_nopunc = 0.0
    lcomplete_match_nopunc = 0.0

    corr_root = 0.0
    total_root = 0.0
    start = 1 if symbolic_root else 0
    end = 1 if symbolic_end else 0
    for i in range(batch_size):
        ucm = 1.0
        lcm = 1.0
        ucm_nopunc = 1.0
        lcm_nopunc = 1.0
        for j in range(start, lengths[i] - end):
            word = word_alphabet.get_instance(words[i, j])
            pos = pos_alphabet.get_instance(postags[i, j])

            total += 1
            if heads[i, j] == heads_pred[i, j]:
                ucorr += 1
                if types[i, j] == types_pred[i, j]:
                    lcorr += 1
                else:
                    lcm = 0
            else:
                ucm = 0
                lcm = 0

            if not is_punctuation(word, pos, punct_set):
                total_nopunc += 1
                if heads[i, j] == heads_pred[i, j]:
                    ucorr_nopunc += 1
                    if types[i, j] == types_pred[i, j]:
                        lcorr_nopunc += 1
                    else:
                        lcm_nopunc = 0
                else:
                    ucm_nopunc = 0
                    lcm_nopunc = 0

            if heads[i, j] == 0:
                total_root += 1
                corr_root += 1 if heads_pred[i, j] == 0 else 0

        ucomplete_match += ucm
        lcomplete_match += lcm
        ucomplete_match_nopunc += ucm_nopunc
        lcomplete_match_nopunc += lcm_nopunc

    return (
        (ucorr, lcorr, total, ucomplete_match, lcomplete_match),
        (
            ucorr_nopunc,
            lcorr_nopunc,
            total_nopunc,
            ucomplete_match_nopunc,
            lcomplete_match_nopunc,
        ),
        (corr_root, total_root),
        batch_size,
    )

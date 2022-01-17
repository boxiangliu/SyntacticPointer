__author__ = "max"

import numpy as np
import torch
from lightning.io.reader import CoNLLXReader
from lightning.io.conllx_data import _buckets, NUM_SYMBOLIC_TAGS, create_alphabets
from lightning.io.common import DIGIT_RE, MAX_CHAR_LENGTH, UNK_ID
from lightning.io.common import (
    PAD_CHAR,
    PAD,
    PAD_POS,
    PAD_TYPE,
    PAD_ID_CHAR,
    PAD_ID_TAG,
    PAD_ID_WORD,
)
from lightning.io.common import (
    ROOT,
    END,
    ROOT_CHAR,
    ROOT_POS,
    ROOT_TYPE,
    END_CHAR,
    END_POS,
    END_TYPE,
)


def _generate_stack_inputs(heads, types):
    stacked_heads = []
    children = [0 for _ in range(len(heads) - 1)]
    stacked_types = []

    position = 1
    for child in range(len(heads)):
        if child == 0:
            continue
        stacked_heads.append(child)
        head = heads[child]
        children[child - 1] = head
        stacked_types.append(types[child])
        position += 1

    return (
        stacked_heads,
        children,
        stacked_types,
    )


def read_data(
    source_path,
    word_alphabet,
    char_alphabet,
    pos_alphabet,
    type_alphabet,
    max_size=None,
    normalize_digits=True,
):
    data = []
    max_length = 0
    max_char_length = 0
    print("Reading data from %s" % source_path)
    counter = 0
    reader = CoNLLXReader(
        source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet
    )
    inst = reader.getNext(
        normalize_digits=normalize_digits, symbolic_root=True, symbolic_end=False
    )
    while inst is not None and (not max_size or counter < max_size):
        counter += 1
        if counter % 10000 == 0:
            print("reading data: %d" % counter)

        sent = inst.sentence

        (
            stacked_heads,
            children,
            stacked_types,
        ) = _generate_stack_inputs(inst.heads, inst.type_ids)

        data.append(
            [
                sent.word_ids,
                sent.char_id_seqs,
                inst.pos_ids,
                inst.heads,
                inst.type_ids,
                stacked_heads,
                children,
                stacked_types,
            ]
        )
        max_len = max([len(char_seq) for char_seq in sent.char_seqs])
        if max_char_length < max_len:
            max_char_length = max_len
        if max_length < inst.length():
            max_length = inst.length()
        inst = reader.getNext(
            normalize_digits=normalize_digits, symbolic_root=True, symbolic_end=False
        )
    reader.close()
    print("Total number of data: %d" % counter)

    data_size = len(data)
    char_length = min(MAX_CHAR_LENGTH, max_char_length)
    wid_inputs = np.empty([data_size, max_length], dtype=np.int64)
    cid_inputs = np.empty([data_size, max_length, char_length], dtype=np.int64)
    pid_inputs = np.empty([data_size, max_length], dtype=np.int64)
    hid_inputs = np.empty([data_size, max_length], dtype=np.int64)
    tid_inputs = np.empty([data_size, max_length], dtype=np.int64)

    masks_e = np.zeros([data_size, max_length], dtype=np.float32)
    single = np.zeros([data_size, max_length], dtype=np.int64)
    lengths = np.empty(data_size, dtype=np.int64)

    stack_hid_inputs = np.empty([data_size, max_length - 1], dtype=np.int64)
    chid_inputs = np.empty([data_size, max_length - 1], dtype=np.int64)
    stack_tid_inputs = np.empty([data_size, max_length - 1], dtype=np.int64)

    masks_d = np.zeros([data_size, max_length - 1], dtype=np.float32)

    for i, inst in enumerate(data):
        (
            wids,
            cid_seqs,
            pids,
            hids,
            tids,
            stack_hids,
            chids,
            stack_tids,
        ) = inst
        inst_size = len(wids)
        lengths[i] = inst_size
        # word ids
        wid_inputs[i, :inst_size] = wids
        wid_inputs[i, inst_size:] = PAD_ID_WORD
        for c, cids in enumerate(cid_seqs):
            cid_inputs[i, c, : len(cids)] = cids
            cid_inputs[i, c, len(cids) :] = PAD_ID_CHAR
        cid_inputs[i, inst_size:, :] = PAD_ID_CHAR
        # pos ids
        pid_inputs[i, :inst_size] = pids
        pid_inputs[i, inst_size:] = PAD_ID_TAG
        # type ids
        tid_inputs[i, :inst_size] = tids
        tid_inputs[i, inst_size:] = PAD_ID_TAG
        # heads
        hid_inputs[i, :inst_size] = hids
        hid_inputs[i, inst_size:] = PAD_ID_TAG
        # masks_e
        masks_e[i, :inst_size] = 1.0
        for j, wid in enumerate(wids):
            if word_alphabet.is_singleton(wid):
                single[i, j] = 1

        # inst_size_decoder = 2 * inst_size - 1
        inst_size_decoder = inst_size - 1
        # stacked heads
        stack_hid_inputs[i, :inst_size_decoder] = stack_hids
        stack_hid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
        # children
        chid_inputs[i, :inst_size_decoder] = chids
        chid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
        # stacked types
        stack_tid_inputs[i, :inst_size_decoder] = stack_tids
        stack_tid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
        # masks_d
        masks_d[i, :inst_size_decoder] = 1.0

    words = torch.from_numpy(wid_inputs)
    chars = torch.from_numpy(cid_inputs)
    pos = torch.from_numpy(pid_inputs)
    heads = torch.from_numpy(hid_inputs)
    types = torch.from_numpy(tid_inputs)
    masks_e = torch.from_numpy(masks_e)
    single = torch.from_numpy(single)
    lengths = torch.from_numpy(lengths)

    stacked_heads = torch.from_numpy(stack_hid_inputs)
    children = torch.from_numpy(chid_inputs)
    stacked_types = torch.from_numpy(stack_tid_inputs)
    masks_d = torch.from_numpy(masks_d)

    data_tensor = {
        "WORD": words,
        "CHAR": chars,
        "POS": pos,
        "HEAD": heads,
        "TYPE": types,
        "MASK_ENC": masks_e,
        "SINGLE": single,
        "LENGTH": lengths,
        "STACK_HEAD": stacked_heads,
        "CHILD": children,
        "STACK_TYPE": stacked_types,
        "MASK_DEC": masks_d,
    }
    return data_tensor, data_size


def read_bucketed_data(
    source_path,
    word_alphabet,
    char_alphabet,
    pos_alphabet,
    type_alphabet,
    max_size=None,
    normalize_digits=True,
):
    data = [[] for _ in _buckets]
    max_char_length = [0 for _ in _buckets]
    print("Reading data from %s" % source_path)
    counter = 0
    reader = CoNLLXReader(
        source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet
    )
    inst = reader.getNext(
        normalize_digits=normalize_digits, symbolic_root=True, symbolic_end=False
    )
    while inst is not None and (not max_size or counter < max_size):
        counter += 1
        if counter % 10000 == 0:
            print("reading data: %d" % counter)

        inst_size = inst.length()
        sent = inst.sentence
        for bucket_id, bucket_size in enumerate(_buckets):
            if inst_size < bucket_size:
                (
                    stacked_heads,
                    children,
                    stacked_types,
                ) = _generate_stack_inputs(inst.heads, inst.type_ids)

                data[bucket_id].append(
                    [
                        sent.word_ids,
                        sent.char_id_seqs,
                        inst.pos_ids,
                        inst.heads,
                        inst.type_ids,
                        stacked_heads,
                        children,
                        stacked_types,
                    ]
                )
                max_len = max([len(char_seq) for char_seq in sent.char_seqs])
                if max_char_length[bucket_id] < max_len:
                    max_char_length[bucket_id] = max_len
                break

        inst = reader.getNext(
            normalize_digits=normalize_digits, symbolic_root=True, symbolic_end=False
        )
    reader.close()
    print("Total number of data: %d" % counter)

    bucket_sizes = [len(data[b]) for b in range(len(_buckets))]
    data_tensors = []
    for bucket_id in range(len(_buckets)):
        bucket_size = bucket_sizes[bucket_id]
        if bucket_size == 0:
            data_tensors.append((1, 1))
            continue

        bucket_length = _buckets[bucket_id]
        char_length = min(MAX_CHAR_LENGTH, max_char_length[bucket_id])
        wid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        cid_inputs = np.empty([bucket_size, bucket_length, char_length], dtype=np.int64)
        pid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        hid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        tid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)

        masks_e = np.zeros([bucket_size, bucket_length], dtype=np.float32)
        single = np.zeros([bucket_size, bucket_length], dtype=np.int64)
        lengths = np.empty(bucket_size, dtype=np.int64)

        stack_hid_inputs = np.empty([bucket_size, bucket_length - 1], dtype=np.int64)
        chid_inputs = np.empty([bucket_size, bucket_length - 1], dtype=np.int64)
        stack_tid_inputs = np.empty([bucket_size, bucket_length - 1], dtype=np.int64)

        masks_d = np.zeros([bucket_size, bucket_length - 1], dtype=np.float32)

        for i, inst in enumerate(data[bucket_id]):
            (
                wids,
                cid_seqs,
                pids,
                hids,
                tids,
                stack_hids,
                chids,
                stack_tids,
            ) = inst
            inst_size = len(wids)
            lengths[i] = inst_size
            # word ids
            wid_inputs[i, :inst_size] = wids
            wid_inputs[i, inst_size:] = PAD_ID_WORD
            for c, cids in enumerate(cid_seqs):
                cid_inputs[i, c, : len(cids)] = cids
                cid_inputs[i, c, len(cids) :] = PAD_ID_CHAR
            cid_inputs[i, inst_size:, :] = PAD_ID_CHAR
            # pos ids
            pid_inputs[i, :inst_size] = pids
            pid_inputs[i, inst_size:] = PAD_ID_TAG
            # type ids
            tid_inputs[i, :inst_size] = tids
            tid_inputs[i, inst_size:] = PAD_ID_TAG
            # heads
            hid_inputs[i, :inst_size] = hids
            hid_inputs[i, inst_size:] = PAD_ID_TAG
            # masks_e
            masks_e[i, :inst_size] = 1.0
            for j, wid in enumerate(wids):
                if word_alphabet.is_singleton(wid):
                    single[i, j] = 1

            # inst_size_decoder = 2 * inst_size - 1
            inst_size_decoder = inst_size - 1
            # stacked heads
            stack_hid_inputs[i, :inst_size_decoder] = stack_hids
            stack_hid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            # children
            chid_inputs[i, :inst_size_decoder] = chids
            chid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            # stacked types
            stack_tid_inputs[i, :inst_size_decoder] = stack_tids
            stack_tid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            # masks_d
            masks_d[i, :inst_size_decoder] = 1.0

        words = torch.from_numpy(wid_inputs)
        chars = torch.from_numpy(cid_inputs)
        pos = torch.from_numpy(pid_inputs)
        heads = torch.from_numpy(hid_inputs)
        types = torch.from_numpy(tid_inputs)
        masks_e = torch.from_numpy(masks_e)
        single = torch.from_numpy(single)
        lengths = torch.from_numpy(lengths)

        stacked_heads = torch.from_numpy(stack_hid_inputs)
        children = torch.from_numpy(chid_inputs)
        stacked_types = torch.from_numpy(stack_tid_inputs)
        masks_d = torch.from_numpy(masks_d)

        data_tensor = {
            "WORD": words,
            "CHAR": chars,
            "POS": pos,
            "HEAD": heads,
            "TYPE": types,
            "MASK_ENC": masks_e,
            "SINGLE": single,
            "LENGTH": lengths,
            "STACK_HEAD": stacked_heads,
            "CHILD": children,
            "STACK_TYPE": stacked_types,
            "MASK_DEC": masks_d,
        }
        data_tensors.append(data_tensor)

    return data_tensors, bucket_sizes

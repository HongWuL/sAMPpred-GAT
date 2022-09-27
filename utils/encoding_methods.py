import os
import numpy as np


def onehot_encoding(seqs):
    residues = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K',
                'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

    encoding_map = np.eye(len(residues))

    residues_map = {}
    for i, r in enumerate(residues):
        residues_map[r] = encoding_map[i]

    res_seqs = []

    for seq in seqs:
        tmp_seq = [residues_map[r] for r in seq]
        res_seqs.append(np.array(tmp_seq))

    return res_seqs

def position_encoding(seqs):
    """
    Position encoding features introduced in "Attention is all your need",
    the b is changed to 1000 for the short length of peptides.
    """
    d = 20
    b = 1000
    res = []
    for seq in seqs:
        N = len(seq)
        value = []
        for pos in range(N):
            tmp = []
            for i in range(d // 2):
                tmp.append(pos / (b ** (2 * i / d)))
            value.append(tmp)
        value = np.array(value)
        pos_encoding = np.zeros((N, d))
        pos_encoding[:, 0::2] = np.sin(value[:, :])
        pos_encoding[:, 1::2] = np.cos(value[:, :])
        res.append(pos_encoding)
    return res


def load_pssm(query, pssm_path):
    """
    :param query: query id
    :param pssm_path: dir saving pssm files
    """
    if pssm_path[-1] != '/': pssm_path += '/'
    with open(pssm_path + query + '.pssm', 'r') as f:
        lines = f.readlines()
        res = []
        for line in lines[3:]:
            line = line.strip()
            lst = line.split(' ')
            while '' in lst:
                lst.remove('')
            if len(lst) == 0:
                break
            r = lst[2:22]
            r = [int(x) for x in r]
            res.append(r)
    return res


def load_hhm(query, hhm_path):
    """
    :param query: query id
    :param hhm_path: dir saving hhm files
    """
    if hhm_path[-1] != '/': hhm_path += '/'
    with open(hhm_path + query + '.hhm', 'r') as f:
        lines = f.readlines()
        res = []
        tag = 0
        for line in lines:
            line = line.strip()
            if line == '#':
                tag = 1
                continue
            if tag != 0 and tag < 5:
                tag += 1
                continue
            if tag >= 5:
                line = line.replace('*', '0')
                lst = line.split('\t')
                if len(lst) >= 20:
                    tmp0 = [int(lst[0].split(' ')[-1])]  # First number
                    tmp1 = list(map(int, lst[1:20]))
                    tmp0.extend(tmp1)
                    normed = [i if i == 0 else 2 ** (-0.001 * i) for i in tmp0]
                    res.append(normed)
    return res


def pssm_encoding(ids, pssm_dir):
    """
    parser pssm features
    """
    if pssm_dir[-1] != '/': pssm_dir += '/'
    pssm_fs = os.listdir(pssm_dir + 'output/')

    res = []
    for id in ids:
        name = id
        if id[0] == '>': name = id[1:]
        if name + '.pssm' in pssm_fs:
            # psiblast
            tmp = load_pssm(name, pssm_dir + 'output/')
            res.append(np.array(tmp))
        else:
            # blosum
            tmp = load_pssm(name, pssm_dir + 'blosum/')
            res.append(np.array(tmp))
    return res


def hhm_encoding(ids, hhm_dir):
    """
    parser pssm features
    """
    if hhm_dir[-1] != '/': hhm_dir += '/'
    hhm_fs = os.listdir(hhm_dir + 'output/')
    res = []
    for id in ids:
        name = id
        if id[0] == '>': name = id[1:]
        assert name + '.hhm' in hhm_fs
        tmp = load_hhm(name, hhm_dir + 'output/')
        res.append(np.array(tmp))

    return res

def add(e1, e2):
    res = []
    for i in range(len(e1)):
        res.append(e1[i] + e2[i])
    return res

def cat(*args):
    """
    :param args: feature matrices
    """
    res = args[0]
    for matrix in args[1:]:
        for i in range(len(matrix)):
            res[i] = np.hstack((res[i], matrix[i]))
    return res

if __name__ == '__main__':
    position_encoding(['ARFGD', 'AAAAAA'])

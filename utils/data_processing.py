import re
import numpy as np
import torch
from utils.encoding_methods import onehot_encoding, pssm_encoding, position_encoding, hhm_encoding,  cat
from torch_geometric.data import Data, DataLoader


def load_seqs(fn, label=1):
    """
    :param fn: source file name in fasta format
    :param tag: label = 1(positive, AMPs) or 0(negative, non-AMPs)
    :return:
        ids: name list
        seqs: peptide sequence list
        labels: label list
    """
    ids = []
    seqs = []
    t = 0
    # Filter out some peptide sequences
    pattern = re.compile('[^ARNDCQEGHILKMFPSTWYV]')
    with open(fn, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line[0] == '>':
                t = line.replace('|', '_')
            elif len(pattern.findall(line)) == 0:
                seqs.append(line)
                ids.append(t)
                t = 0
    if label == 1:
        labels = np.ones(len(ids))
    else:
        labels = np.zeros(len(ids))
    return ids, seqs, labels


def load_data(fasta_path, npz_dir, threshold=37, label=1, add_self_loop=True):
    """
    :param fasta_path: file path of fasta
    :param npz_dir: dir that saves npz files
    :param threshold: threshold for build adjacency matrix
    :param label: labels
    :return:
        data_list: list of Data
        labels: list of labels
    """
    ids, seqs, labels = load_seqs(fasta_path, label)
    As, Es = get_cmap(npz_dir, ids, threshold, add_self_loop)

    one_hot_encodings = onehot_encoding(seqs)
    position_encodings = position_encoding(seqs)
    pssm_dir = '/'.join(fasta_path.split('/')[:-1]) + '/pssm/'
    pssm_encodings = pssm_encoding(ids, pssm_dir)

    hhm_dir = '/'.join(fasta_path.split('/')[:-1]) + '/hhm/'
    hhm_encodings = hhm_encoding(ids, hhm_dir)

    Xs = cat(one_hot_encodings, position_encodings, pssm_encodings, hhm_encodings)

    n_samples = len(As)
    data_list = []
    for i in range(n_samples):
        data_list.append(to_parse_matrix(As[i], Xs[i], Es[i], labels[i]))
    return data_list, labels


def get_cmap(npz_folder, ids, threshold, add_self_loop=True):
    if npz_folder[-1] != '/':
        npz_folder += '/'

    list_A = []
    list_E = []

    for id in ids:
        npz = id[1:] + '.npz'
        f = np.load(npz_folder + npz)

        mat_dist = f['dist']
        mat_omega = f['omega']
        mat_theta = f['theta']
        mat_phi = f['phi']

        """ 
        The distance range (2 to 20 Å) is binned into 36 equally spaced segments, 0.5 Å each, 
        plus one bin indicating that residues are not in contact.
            - Improved protein structure prediction using predicted interresidue orientations: 
        """
        dist = np.argmax(mat_dist, axis=2)  # 37 equally spaced segments
        omega = np.argmax(mat_omega, axis=2)
        theta = np.argmax(mat_theta, axis=2)
        phi = np.argmax(mat_phi, axis=2)

        A = np.zeros(dist.shape, dtype=np.int)

        A[dist < threshold] = 1
        A[dist == 0] = 0
        # A[omega < threshold] = 1
        if add_self_loop:
            A[np.eye(A.shape[0]) == 1] = 1
        else:
            A[np.eye(A.shape[0]) == 1] = 0

        dist[A == 0] = 0
        omega[A == 0] = 0
        theta[A == 0] = 0
        phi[A == 0] = 0

        dist = np.expand_dims(dist, -1)
        omega = np.expand_dims(omega, -1)
        theta = np.expand_dims(theta, -1)
        phi = np.expand_dims(phi, -1)

        edges = dist
        edges = np.concatenate((edges, omega), axis=-1)
        edges = np.concatenate((edges, theta), axis=-1)
        edges = np.concatenate((edges, phi), axis=-1)

        list_A.append(A)
        list_E.append(edges)

    return list_A, list_E


def to_parse_matrix(A, X, E, Y, eps=1e-6):
    """
    :param A: Adjacency matrix with shape (n_nodes, n_nodes)
    :param E: Edge matrix with shape (n_nodes, n_nodes, n_edge_features)
    :param X: node embedding with shape (n_nodes, n_node_features)
    :return:
    """
    num_row, num_col = A.shape
    rows = []
    cols = []
    e_vec = []

    for i in range(num_row):
        for j in range(num_col):
            if A[i][j] >= eps:
                rows.append(i)
                cols.append(j)
                e_vec.append(E[i][j])
    edge_index = torch.tensor([rows, cols], dtype=torch.int64)
    x = torch.tensor(X, dtype=torch.float32)
    edge_attr = torch.tensor(e_vec, dtype=torch.float32)
    y = torch.tensor([Y], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)




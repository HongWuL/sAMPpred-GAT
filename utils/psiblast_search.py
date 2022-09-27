# from analysis.data_filter import load_seqs
import os
import shutil
import sys
import pickle as pkl
from tqdm import tqdm, trange
import argparse

def check(names, target_folder):
    processed = os.listdir(target_folder)
    processed = [p[:-5] for p in processed]

    rest_names = []
    for i in range(len(names)):
        name = names[i]
        fname = name.replace('|', '_')[1:]
        if fname not in processed:
            rest_names.append(fname)
    return rest_names

def run(psi, input_fasta, target_folder, db, nr = None):
    """
    Using psiblast to generate .pssm files
    parameters:
        :param psi: path of psiblast software, xxxx/xx/psiblast
        :param input_fasta: input .fasta file containing multiple sequences
        :param target_folder: target output folder, which contains tmp, output, xml, blosum folders
        :param db: nr database
        :param nr: Whether to use the NR database, if so, enter the path of NR database
    """

    tmp_folder = target_folder + 'tmp/'
    pssm_folder = target_folder + 'output/'
    xml_folder = target_folder + 'xml/'

    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)

    if not os.path.exists(pssm_folder):
        os.makedirs(pssm_folder)

    if not os.path.exists(xml_folder):
        os.makedirs(xml_folder)

    # split fasta
    names = []
    seqs = []
    with open(input_fasta, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line[0] == '>':
                if '|' in lines: line = line.replace('|', '_')
                names.append(line)
            else:
                seqs.append(line)

    for i in range(len(names)):
        name = names[i]
        fname = name.replace('|', '_')[1:]
        seq = seqs[i]
        with open(tmp_folder + fname + '.fasta', 'w') as f:
            f.write(name + '\n')
            f.write(seq)

    # nrdb90
    for i in trange(len(names)):
        name = names[i]
        fname = name.replace('|', '_')[1:]
        src = tmp_folder + fname + '.fasta'
        gen_pssm_by_blast(psi, src, pssm_folder, xml_folder, db)

    # NR
    rest_names = check(names, pssm_folder)
    # print('nrdb90:', len(names) - len(rest_names))

    if nr is not None:
        for i in trange(len(rest_names)):
            fname = rest_names[i]
            src = tmp_folder + fname + '.fasta'
            gen_pssm_by_blast(psi, src, pssm_folder, xml_folder, nr)

    # blosum
    rest_names = check(names, pssm_folder)
    # print('blosum:', len(rest_names))

    for i in range(len(names)):
        name = names[i][1:]
        if name in rest_names:
            gen_pssm_by_blosum(seqs[i], 'utils/psiblast/blosum62.pkl', pssm_folder + name + '.pssm')

    # remove tmp and xml
    shutil.rmtree(tmp_folder)
    shutil.rmtree(xml_folder)


def gen_pssm_by_blast(psi, src, pssm_folder, xml_folder, db):
    """
    """
    name = src.split('/')[-1][:-6]

    output = pssm_folder + name + '.pssm'

    xml_file = xml_folder + name + '.xml'

    psiblast_cmd = psi

    evalue_threshold = 0.01
    num_iter = 3
    outfmt_type = 5

    cmd = ' '.join([psiblast_cmd,
                   '-query ' + src,
                   '-db ' + db,
                    '-out ' + xml_file,
                   '-evalue ' + str(evalue_threshold),
                    '-num_iterations ' + str(num_iter),
                   '-num_threads ' + '6',
                    '-out_ascii_pssm ' + output,
                    '-outfmt ' + str(outfmt_type)
                    ]
                )
    os.system(cmd)

def read_blosum(blosum_dir):
    """Read blosum dict and delete some keys and values."""
    with open(blosum_dir, 'rb') as f:
        blosum_dict = pkl.load(f)

    blosum_dict.pop('*')
    blosum_dict.pop('B')
    blosum_dict.pop('Z')
    blosum_dict.pop('X')
    blosum_dict.pop('alphas')

    for key in blosum_dict:
        for i in range(4):
            blosum_dict[key].pop()
    return blosum_dict

def gen_pssm_by_blosum(seq, blosum_dir, trg):

    blosum = read_blosum(blosum_dir)
    enc = []
    for aa in seq:
        enc.append(blosum[aa])
    with open(trg, 'w') as f:
        for i in range(3): f.write('\n')
        for i, s in enumerate(seq):
            str_list = map(str, enc[i])
            f.write(' ' + str(i) + ' ' + s + ' ')
            f.write(' '.join(str_list))
            f.write('\n')



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-q', type=str)
    parser.add_argument('-o', type=str)
    parser.add_argument('-d',  type=str)

    # args = parser.parse_args()
    # ids, seqs = load_seqs(args.q)
    # gen_pssm(ids, seqs, args.o, args.d)


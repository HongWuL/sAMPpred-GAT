import os
import argparse
from tqdm.std import trange


def run(input, tg, tg_hhm, tmp_folder, db):
    """
    Using HHblits to search against Uniclust2018 database, to generate .a3m and .hhm files
    parameters:
        :param input: input .fasta file containing multiple sequences
        :param tg: target output .a3m folder
        :param tg_hhm: target output .hhm folder
        :param tmp_folder: tmp folder saving the .fasta files containing a singe sequence that is split from the input files.
        :param db: the path of Uniclust2018
    """

    names = []
    seqs = []
    with open(input, 'r') as f:
        lines = f.readlines()
        print("Length:", len(lines) / 2)
        for line in lines:
            line = line.strip()
            if line[0] == '>':
                names.append(line)
            else:
                seqs.append(line)

    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)

    for i in range(len(names)):
        name = names[i]
        fname = name.replace('|', '_')[1:]
        seq = seqs[i]
        with open(tmp_folder + fname + '.fasta', 'w') as f:
            f.write(name + '\n')
            f.write(seq)

    if not os.path.exists(tg):
        os.makedirs(tg)
    if not os.path.exists(tg_hhm):
        os.makedirs(tg_hhm)

    try:
        for i in trange(len(names)):
            name = names[i]
            fname = name.replace('|', '_')[1:]
            fn = tmp_folder + fname + '.fasta'
            cmd = 'hhblits -i ' + fn + ' -o ' + tg + 'tmp.hhr' + \
                  ' -oa3m ' + tg + fname + '.a3m -ohhm ' + tg_hhm + fname + '.hhm' + \
                  ' -d ' + db + ' -cpu 8 -v 0 -n 3 -e 0.01'
            os.system(cmd)
    except:
        print('Failed to search !')
    finally:
        # remove temp folder
        for f in os.listdir(tmp_folder):
            os.remove(tmp_folder + f)


def main(args):
    input = args.i
    pt = args.pt
    db = args.d
    tg = args.oa3m
    tg_hhm = args.ohhm

    tmp_folder = './tmp/' + pt + '/'

    names = []
    seqs = []
    with open(input, 'r') as f:
        lines = f.readlines()
        print("Length:", len(lines) / 2)
        for line in lines:
            line = line.strip()
            if line[0] == '>':
                names.append(line)
            else:
                seqs.append(line)

    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)

    for i in range(len(names)):
        name = names[i]
        fname = name.replace('|', '_')[1:]
        seq = seqs[i]
        with open(tmp_folder + fname + '.fasta', 'w') as f:
            f.write(name + '\n')
            f.write(seq)

    if not os.path.exists(tg + pt):
        os.makedirs(tg + pt)
    if not os.path.exists(tg + pt):
        os.makedirs(tg_hhm + pt)

    try:
        for i in trange(len(names)):
            name = names[i]
            fname = name.replace('|', '_')[1:]
            fn = tmp_folder + fname + '.fasta'
            cmd = 'hhblits -i ' + fn + ' -o ' + tg + 'tmp.hhr' + \
                  ' -oa3m ' + tg + pt + '/' + fname + '.a3m -ohhm ' + tg_hhm + pt + '/' + fname + '.hhm' + \
                  ' -d ' + db + ' -cpu 8 -v 0 -n 3 -e 0.01'
            os.system(cmd)
    except:
        print('Failed to search !')
    finally:
        # remove temp folder
        for f in os.listdir(tmp_folder):
            os.remove(tmp_folder + f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pt', type=str, default='AMP', help='Peptide type, used for name the output files')
    parser.add_argument('-d', type=str, help='Database that is searched against')
    parser.add_argument('-i', type=str, default='datasets/input.fasta', help='Input files in fasta format')
    parser.add_argument('-oa3m', type=str, default='result_a3m/', help='Output folder saving o3m files')
    parser.add_argument('-ohhm', type=str, default='result_hhm/', help='Output folder saving hhm files')
    args = parser.parse_args()
    main(args)
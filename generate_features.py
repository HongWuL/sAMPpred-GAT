import os
import utils.hhblits_search as hh
import utils.psiblast_search as psi
import yaml
import argparse

# Load the paths of tools and databases 
with open("config.yaml", 'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

psiblast = cfg['psiblast']
hhblits = cfg['hhblits']
rosetta = cfg['rosetta']

# Databases and model
nrdb90 = cfg['nrdb90']
nr = cfg['nr']
uniclust = cfg['uniclust']
rosetta_model = cfg['rosetta_model']


def generate_features(args):
    """
    """
    feas = args.feas
    if 'HHM' in feas:
        hh.run(hhblits, args.hhm_ifasta, args.hhm_oa3m, args.hhm_ohhm, args.hhm_tmp, uniclust)

    if 'NPZ' in feas:
        rosetta_cmd = 'python ' + rosetta + ' ' + \
                      args.tr_ia3m + ' ' + args.tr_onpz + ' -m ' + rosetta_model
        os.system(rosetta_cmd)

    if 'PSSM' in feas:
        psi.run(psiblast, args.pssm_ifasta, args.pssm_opssm, nrdb90, nr)

if __name__ == '__main__':
    # generate contact map, pssm and hhm features before train and test model.
    parser = argparse.ArgumentParser()
    parser.add_argument('-feas', type=str, nargs='+', default=['PSSM', 'HHM', 'NPZ'], help='Feature names')

    # HHblits parameters
    parser.add_argument('-hhm_ifasta', type=str, default='example/test.fasta',
                        help='Input a file with fasta format for hhblits search')
    parser.add_argument('-hhm_oa3m', type=str, default='example/a3m/',
                        help='Output folder saving .a3m files')
    parser.add_argument('-hhm_ohhm', type=str, default='example/hhm/',
                        help='Output folder saving .hhm files')
    parser.add_argument('-hhm_tmp', type=str, default='example/tmp/', help='Temp folder')


    # trRosetta parameters
    parser.add_argument('-tr_ia3m', type=str, default='example/a3m/',
                        help='Input folder saving .a3m files')
    parser.add_argument('-tr_onpz', type=str, default='example/npz/',
                        help='Output folder saving .npz files')

    # PSSM parameters
    parser.add_argument('-pssm_ifasta', type=str, default='example/test.fasta', help='Input .fasta file for psiblast search')
    parser.add_argument('-pssm_opssm', type=str, default='example/pssm/', help='Output folder saving .pssm files')

    args = parser.parse_args()

    generate_features(args)

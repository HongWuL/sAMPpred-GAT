import os
import utils.hhblits_search as hh
import utils.psiblast_search as psi

import argparse

def generate_features(args):
    """
    """
    feas = args.feas
    if 'HHM' in feas:
        hh.run(args.hhm_ifasta, args.hhm_oa3m, args.hhm_ohhm, args.hhm_tmp, args.hhm_db)

    if 'NPZ' in feas:
        rosetta_cmd = 'python utils/trRosetta/predict_many.py ' + \
                      args.tr_ia3m + ' ' + args.tr_onpz + ' -m ' + args.tr_model
        os.system(rosetta_cmd)

    if 'PSSM' in feas:
        psi.run(args.pssm_ifasta, args.pssm_opssm, args.pssm_db, args.pssm_nr)

if __name__ == '__main__':
    # generate contact map, pssm and hhm features before train and test model.
    parser = argparse.ArgumentParser()
    parser.add_argument('-feas', type=str, default=['PSSM', 'HHM', 'NPZ'], help='Feature names')

    # HHblits parameters
    parser.add_argument('-hhm_ifasta', type=str, default='test_feature_generation/test.fasta',
                        help='Input a file with fasta format for hhblits search')
    parser.add_argument('-hhm_oa3m', type=str, default='test_feature_generation/a3m/',
                        help='Output folder saving .a3m files')
    parser.add_argument('-hhm_ohhm', type=str, default='test_feature_generation/hhm/',
                        help='Output folder saving .hhm files')
    parser.add_argument('-hhm_tmp', type=str, default='test_feature_generation/tmp/', help='Temp folder')
    parser.add_argument('-hhm_db', type=str, default='uniclust30_2018_08/uniclust30_2018_08',
                        help='Uniclust database for hhblits')

    # trRosetta parameters
    parser.add_argument('-tr_model', type=str, default='utils/trRosetta/model2019_07',
                        help='Pretrained trRosetta model')
    parser.add_argument('-tr_ia3m', type=str, default='test_feature_generation/a3m/',
                        help='Input folder saving .a3m files')
    parser.add_argument('-tr_onpz', type=str, default='test_feature_generation/npz/',
                        help='Output folder saving .npz files')

    # PSSM parameters
    parser.add_argument('-pssm_ifasta', type=str, default='test_feature_generation/test.fasta', help='Input .fasta file for psiblast search')
    parser.add_argument('-pssm_opssm', type=str, default='test_feature_generation/pssm/', help='Output folder saving .pssm files')
    parser.add_argument('-pssm_db', type=str, default='utils/psiblast/nrdb90/nrdb90', help='Nrdb90 database for psiblast')
    parser.add_argument('-pssm_nr', type=str, default='utils/psiblast/nr/nr', help='NR database for psiblast, if None, only nrdb90 and blosum are used')

    args = parser.parse_args()

    generate_features(args)

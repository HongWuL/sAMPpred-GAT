#! /bin/bash

pos=example/positive
neg=example/negative

python generate_features.py -hhm_ifasta $pos/example_pos.fasta -hhm_oa3m $pos/a3m/ -hhm_ohhm $pos/hhm/ -hhm_tmp $pos/tmp/ -tr_ia3m $pos/a3m/ -tr_onpz $pos/npz/ -pssm_ifasta $pos/example_pos.fasta -pssm_opssm $pos/pssm/

python generate_features.py -hhm_ifasta $neg/example_neg.fasta -hhm_oa3m $neg/a3m/ -hhm_ohhm $neg/hhm/ -hhm_tmp $neg/tmp/ -tr_ia3m $neg/a3m/ -tr_onpz $neg/npz/ -pssm_ifasta $neg/example_neg.fasta -pssm_opssm $neg/pssm/


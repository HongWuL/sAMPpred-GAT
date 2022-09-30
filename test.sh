#! /bin/bash

pos=data/test_data/positive
neg=data/test_data/negative

mkdir -p $pos 
mkdir -p $neg

cp datasets/independent\ test\ datasets/XUAMP/XU_AMP.fasta $pos/XU_AMP.fasta
cp datasets/independent\ test\ datasets/XUAMP/XU_nonAMP.fasta $neg/XU_nonAMP.fasta

python generate_features.py -hhm_ifasta $pos/XU_AMP.fasta -hhm_oa3m $pos/a3m/ -hhm_ohhm $pos/hhm/ -hhm_tmp $pos/tmp/ -tr_ia3m $pos/a3m/ -tr_onpz $pos/npz/ -pssm_ifasta $pos/XU_AMP.fasta -pssm_opssm $pos/pssm/
python generate_features.py -hhm_ifasta $neg/XU_nonAMP.fasta -hhm_oa3m $neg/a3m/ -hhm_ohhm $neg/hhm/ -hhm_tmp $neg/tmp/ -tr_ia3m $neg/a3m/ -tr_onpz $neg/npz/ -pssm_ifasta $neg/XU_nonAMP.fasta -pssm_opssm $neg/pssm/

python test.py  -pos_t $pos/XU_AMP.fasta -pos_npz $pos/npz/ -neg_t $neg/XU_nonAMP.fasta -neg_npz $neg/npz/ \
                -save saved_models/samp.model

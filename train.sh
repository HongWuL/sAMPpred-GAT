#! /bin/bash

pos=data/train_data/positive
neg=data/train_data/negative

# python generate_features.py -hhm_ifasta $pos/XU_pretrain_train_positive.fasta -hhm_oa3m $pos/a3m/ -hhm_ohhm $pos/hhm/ -hhm_tmp $pos/tmp/ -tr_ia3m $pos/a3m/ -tr_onpz $pos/npz/ -pssm_ifasta $pos/example_pos.fasta -pssm_opssm $pos/pssm/
# python generate_features.py -hhm_ifasta $neg/XU_pretrain_train_negative.fasta -hhm_oa3m $neg/a3m/ -hhm_ohhm $neg/hhm/ -hhm_tmp $neg/tmp/ -tr_ia3m $neg/a3m/ -tr_onpz $neg/npz/ -pssm_ifasta $neg/example_neg.fasta -pssm_opssm $neg/pssm/

python train.py -pos_t $pos/XU_pretrain_train_positive.fasta -pos_v $pos/XU_pretrain_val_positive.fasta -pos_npz $pos/npz/ \
                -neg_t $neg/XU_pretrain_train_negative.fasta -neg_v $neg/XU_pretrain_val_negative.fasta -neg_npz $neg/npz/ \
                -save saved_models/XU.model

python train.py -pos_t $pos/XU_train_positive.fasta -pos_v $pos/XU_val_positive.fasta -pos_npz $pos/npz/ \
                -neg_t $neg/XU_train_negative.fasta -neg_v $neg/XU_val_negative.fasta -neg_npz $neg/npz/ \
                -lr 0.0001 -e 20 -pretrained_model saved_models/auc_XU.model -save saved_models/XU_final.model

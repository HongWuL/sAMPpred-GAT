# sAMPpred-GAT
The implementation of the paper ***sAMPpred-GAT: Prediction of Antimicrobial Peptide by Graph Attention Network and Predicted Peptide Structure***

## Requirements
The majoy dependencies used in this project are as following:

```
python 3.7              
scikit-learn  0.24.2
torch 1.9.0+cu111
torch-cluster 1.5.9
torch-geometric 1.7.2
torch-scatter 2.0.8
torch-sparse  0.6.11
tensorboard 2.6.0
matplotlib  3.4.3                                                                                                                          
networkx  2.5.1                                                                                                                          
numpy 1.20.3
tqdm  4.62.2
```
Besides, two multiple sequence alignment tools and three databases  are required:
Tools:
```
psi-blast 2.12.0
hhblits 3.3.0
```
The installation method and location:
hhblits:
```
conda install -c conda-forge -c bioconda hhsuite==3.3.0
```
psibliast:
```
wget ftp://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.12.0/ncbi-blast-2.12.0+-x64-linux.tar.gz
```

Databases:
```
nrdb90(http://ftp.ebi.ac.uk/pub/databases/nrdb90/)
NR(https://ftp.ncbi.nlm.nih.gov/blast/db/)
uniclust30_2018_08(https://wwwuser.gwdg.de/~compbiol/uniclust/2018_08/)
```
The structures are predicted by trRosetta(https://github.com/gjoni/trRosetta), you need to download and place the trRosetta pretrain model(model2019_07) in directory uitls/trRosetta/. The dependecies used by trRosetta are also required. 

## Feature extraction

`generate_features.py` is the entry of feature extraction process. An usage example:
```
python generate_features.py -feas PSSM HHM NPZ -hhm_ifasta test_feature_generation/test.fasta -pssm_ifasta test_feature_generation/test.fasta
```
Note that, before runing the feature extracting program, the path of psiblast and hhblits in `utils/psiblast_search.py` and `utils/hhblits_search.py` should be checked and replaced into your local path.

## train and test

`train.py` and `test.py` are used for training and testing, respectively. The input folder should like:
```

-positive/
XXX(name of the positive file).fasta
--pssm/
---output/
----A.pssm
----B.pssm
---- ...
--hhm/
---output/
----A.hhm
----B.hhm
---- ...
--npz/
---A.npz
---B.npz

-negative
XXX(name of the negative file).fasta
 --pssm/
---output/
----C.pssm
----D.pssm
---- ...
--hhm/
---output/
----C.hhm
----D.hhm
---- ...
--npz/
---C.npz
---D.npz

```
A trained model for XUAMP is supplied in saved_models/samp.model. 

Note that the `fasta_path_positive`, `npz_dir_positive`, `fasta_path_negative` and `npz_dir_negative` should be replaced as the path of the input folder.





# sAMPpred-GAT
The implementation of the paper ***sAMPpred-GAT: Prediction of Antimicrobial Peptide by Graph Attention Network and Predicted Peptide Structure***

## Requirements
The majoy dependencies used in this project are as following:

```
python 3.7              
scikit-learn  1.0.2
torch 1.11.0+cu113
torch-cluster 1.6.0
torch-geometric 1.7.3
torch-scatter 2.0.9
torch-sparse  0.6.15
tensorflow  1.14.0
tensorboardX 2.5.1
networkx  2.61.3                                                                                                                          
numpy 1.21.6
tqdm  4.64.1
```
For more detailed library information, please refer to `requirements.txt`. 

## Tools
Two multiple sequence alignment tools and three databases are required: 
```
psi-blast 2.12.0
hhblits 3.3.0
```
Databases:
```
nrdb90(http://ftp.ebi.ac.uk/pub/databases/nrdb90/)
NR(https://ftp.ncbi.nlm.nih.gov/blast/db/)
uniclust30_2018_08(https://wwwuser.gwdg.de/~compbiol/uniclust/2018_08/)
```
The structures are predicted by trRosetta(https://github.com/gjoni/trRosetta), you need to download and place the trRosetta pretrain model(model2019_07) as well.

All the defalut paths of the tools and databases are shown in `config.yaml`. `psi-blast` and `hhblist` are recommended to be configured as the system envirenment path. The trRosetta model and the databases are recommended to be placed into the `utils/` directory as desribed in `config.yaml`.
Note that you can change the paths of the tools and databases by configuring `config.yaml` as you need.

## Feature extraction

`generate_features.py` is the entry of feature extraction process. An usage example is shown in `generate_features_example.sh`. 

Run the example by: 
```
chmod +x 777 generate_features_example.sh
./generate_features_example.sh
```
If you want generate the features using your own file in fasta format, just follow the `generate_features_example.sh` and change the pathes as yours.

## train and test

`train.py` and `test.py` are used for training and testing, respectively. 

The input folder should like:

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
Note that `generate_features_example.sh` just generated the corrent folder structure. It is easy to  follow the example to generate the input folder.

### Test 

A trained model for XUAMP is supplied in saved_models/samp.model as an example. Run `test.py` to predict the example sequences:
`python test.py`

If you want test the specific dataset, for example XUAMP, you should copy the files in fasta format in `datasets/independent/` directory into the `data/test_data/positive/` folder and `data/test_data/negative/`, and change the args relative to the input in `test.py`. An example is given in `test.sh`.

### Train

If you want train a model based on the specific dataset, for example XUAMP, you should copy the files in fasta format in `datasets/train datasets/` directory into the `data/train_data/positive/` folder and `data/train_data/negative/`, and change the args relative to the input in `train.py` as the test process. An example is given in `train.sh`.










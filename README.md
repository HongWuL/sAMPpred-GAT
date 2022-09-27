# sAMPpred-GAT
The implementation of the paper ***sAMPpred-GAT: Prediction of Antimicrobial Peptide by Graph Attention Network and Predicted Peptide Structure***

## Requirements
The majoy dependencies used in this project are as following:

```
python  3.7
numpy 1.21.6
pandas  1.3.5
tqdm  4.64.1
pyyaml  6.0
scikit-learn  1.0.2
torch  1.11.0+cu113
torch-cluster  1.6.0
torch-scatter  2.0.9
torch-sparse  0.6.15
torch-geometric  1.7.2
tensorflow  1.14.0
tensorboardX  2.5.1
```
More detailed python libraries used in this project are referred to `requirements.txt`. 
Check your CPU device and install the pytorch and pyG (torch-cluster, torch-scatter, torch-sparse, torch-geometric) according to your CUDA version.
> Note that torch-geometric 1.7.2 and tensorflow 1.14.0 are recommended, becuase our trained model does not support the `torch-geometric` with higher version , and the model from trRosetta does not support the `tensorflow` with higher version.

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


The trRosetta model and the databases are recommended to be placed into the `utils/` directory as desribed in `config.yaml`.
**Note** that you can change the paths of the tools and databases by configuring `config.yaml` as you need.

All the defalut paths of the tools and databases are shown in `config.yaml`. `psi-blast` and `hhblist` are recommended to be configured as the system envirenment path.
Your can follow these steps to install them:
### psiblast

Download 

```
wget ftp://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.12.0/ncbi-blast-2.12.0+-x64-linux.tar.gz
tar zxvf ncbi-blast-2.12.0+-x64-linux.tar.gz
```

Add the path to system envirenment in `~/.bashrc`.

```
export BLAST_HOME={your_path}/ncbi-blast-2.12.0+
export PATH=$PATH:$BLAST_HOME/bin
```

Finally, reload the system envirenment and check the psiblast command:

```
source ~/.bashrc
psiblast -h
```


### hhblits
You can download and install the hhblits througth `conda` quickly.

```
conda install -c conda-forge -c bioconda hhsuite==3.3.0
```
Check the installation:

```
hhblits -h
```

## Feature extraction

`generate_features.py` is the entry of feature extraction process. An usage example is shown in `generate_features_example.sh`. 

Run the example by: 
```
chmod +x generate_features_example.sh
./generate_features_example.sh
```
The features used in this study will be genrerated if your tools and databases are configured correctly.
If you want generate the features using your own file in fasta format, just follow the `generate_features_example.sh` and change the pathes into yours.

## Usage

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
Note that `generate_features_example.sh` just generated the corrent folder structure. Just follow the example to generate the input folder.

### Test 

A trained model for XUAMP is supplied in saved_models/samp.model as an example. Run `test.py` to predict the example sequences:
`python test.py`

If you want test the specific dataset, for example XUAMP, you should copy the corresponding files in fasta format in `datasets/independent test datasets/` directory into the `data/test_data/positive/` and `data/test_data/negative/`, and set the ***args*** relative to the inputs. An example is given in `test.sh`.

### Train

If you want train a model based on the specific dataset, for example XUAMP, you should copy the files in fasta format in `datasets/train datasets/` directory into the `data/train_data/positive/` and `data/train_data/negative/`, and set the ***args*** relative to the inputs. An example is given in `train.sh`.










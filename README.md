## FedNLP

An open-source federated learning framework for FULLY privacy-preserving biomedical data analysis

## First stable release

### What we have :star2: 

- [x] support partially homomorphic encryption (PHE), currently support CPU multi-thread acceleration
- [x] provide flexible FL API that allows cutomized FL algorithms
- [x] support tensorboard for metrics monitoring

### What is expected to see in the next release :rocket: 
- [ ] PHE with CUDA acceleration
- [ ] Fully homomorphic encryption (FHE)
- [ ] Personalized FL algorithms


## Installation
```bash
git clone git@github.com:PL97/BioPrivFL.git #
cd BioPrivFL/

## install necessary dependencies
# for more details about phe -> https://python-paillier.readthedocs.io/en/develop/installation.html
pip install phe
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

```

## Datasets
check demo_FL.py for details

## Usage

```bash
# standard fedavg
python demo_FL.py

# fedavg with phe
python demo_FL.py --phe 

```

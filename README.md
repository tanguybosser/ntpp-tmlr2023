# Large Scale Experimental Study of Neural Temporal Point Processes.

This repository contains the base code implemented in Pytorch for the experiments conducted in the paper ["On the Predictive accuracy of Neural Temporal Point Process Models for Continuous-time Event Data"](https://openreview.net/forum?id=3OSISBQPrM&referrer=%5BTMLR%5D(%2Fgroup%3Fid%3DTMLR)), Tanguy Bosser and Souhaib Ben Taieb, TMLR 2023. 

The base code is heavily built on the implementation of ["Neural Temporal Point Processes For Modeling Electronic Health Records"](https://github.com/babylonhealth/neuralTPPs), Joseph Enguehard, Dan Busbridge, Adam Bozson, Claire Woodcock, and Nils Y. Hammerla. We thank the authors for sharing their valuable code. 

## Contact
+ Tanguy Bosser [tanguy.bosser@umons.ac.be](mailto:tanguy.bosser@umons.ac.be)

## Development
### Installation

In order to use this repository, you need to create an environment that contains [tick](https://github.com/X-DataInitiative/tick). Prior to installing tick, you'll need to have the [Swig](https://swig.org/svn.html) library installed on your device. Refer to the [documentation](https://www.swig.org/Doc4.0/SWIGDocumentation.html#Preface_installation) (in section 1.12) for a step-by-step installation. Once Swig is successfully installed, run the following command to setup the environment:

```shell script
conda env create -f environment.yml
conda activate ntpp_env
```

## Usage

### Training a model

An example of command to train the LogNormalMixture decoder with a GRU encoder is provided below:

```
python3 -u scripts/train.py --dataset 'lastfm_filtered' --load-from-dir 'data' \
--save-results-dir 'tests/lastfm_filtered' --save-check-dir 'checkpoints/tests/lastfm_filtered' \
--eval-metrics True --include-poisson False --patience 100 --batch-size 8 --split 0 \
--encoder 'gru' --encoder-encoding 'temporal_with_labels' --encoder-emb-dim 8 \
--encoder-units-rnn 32 --encoder-layers-rnn 1 \
--encoder-units-mlp 16 --encoder-activation-mlp 'relu' \
--decoder 'log-normal-mixture' \
--decoder-units-mlp 16 --decoder-units-mlp 16 \
--decoder-n-mixture 32 
```

Check the `examples.job` file for more examples. 

According to where you want the results and checkpoints to be saved, you will need to change the "--save-results-dir" and "--save-check-dir" arguments. Additionaly, depending on where you decide to store the datasets, the "--load-from-dir" path needs to be adjusted. 


 To get the complete list of arguments, run the following command:

```
python3 scripts/train.py -h
```

## Data
The preprocessed datasets, as well as the splits divisions, can be found at [this link](https://www.dropbox.com/sh/maq7nju7v5020kp/AABicAxAdkjpn2nvsCxzFBE6a?dl=0). Place the 'data' folder (located within the 'processed' folder) at the top level of this repository to run the commands of the `examples.job` file.  

## Cite
Please cite our paper if you use this code in your own work:
```
@article{bosser2023tmlrneuraltpps,
    title={On the Predictive accuracy of Neural Temporal Point Process Models for Continuous-time Event Data},
    author={Tanguy Bosser and Souhaib Ben Taieb},
    year={2023},
    journal={Transaction of Machine Learning Research (TMLR)}
}
```

## Repo notes
This repository was originally forked from https://github.com/babylonhealth/neuralTPPs

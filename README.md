# Configurational Polymer Fingerprints for Machine Learning

Code for the paper [Configurational Polymer Fingerprints for Machine Learning](https://arxiv.org/abs/2311.14744) by Ishan Kumar, Prateek K. Jha

![Image](https://github.com/Ishan-Kumar2/configurational-polymer-fingerprint/blob/main/polymerfingerprint.png)

## Highlights of the work
* Machine learning (ML) model is developed for a coarse-grained, bead-spring model of polymers. 
* ML model is trained using Monte Carlo (MC) simulations performed on the bead-spring model. 
* Use of both calculated (geometric) and learnt descriptors add value to the ML model. 
* Probability of occurrence of configurations at equilibrium are predicted well by the ML model. 


There are two parts to the code. The first involves C++ code to run Monte Carlo simulation in order to create the Dataset of fingerprints and descriptors. The second part is the python ML Code which uses the dataset from the previous step to train the Autoencoder and the prediction model.

# Usage
## Monte Carlo Simulation 
First step is to compile the C++ code 
This requires the Spectra and Eigen libraries to be present. They can be installed using 
```bash
git clone https://gitlab.com/libeigen/eigen.git
git clone https://github.com/yixuan/spectra
```
Also the [boost](https://www.boost.org/) and [GSL](https://coral.ise.lehigh.edu/jild13/2016/07/11/hello/) libraries need to be installed.

```bash
g++ -I /path/to/Eigen -I /path/to/spectra/include -c montecarlo.cpp objutils.cpp utils.cpp vars.h main.cpp
```
Since the compiled files are also provided, you can directly run
```bash
g++ -o run main.o montecarlo.o objutils.o utils.o -lgsl -lboost_program_options
```


Then the compiled file can be used to create the dataset using the Dataset.sh code as
```bash
./Dataset.sh > output.txt
```

The Dataset creates a folder containing a seperate folders for each run. In order to convert it into a single folder (so that it is easier to read in the ML Dataloader) the `ML/single_folder.py` script can be used (by changing the path variables in the code) as follows.
```bash
python ./ML/single_folder.py
```



## Machine Learning Model 
For training the ML model (Autoencoder) to generate the learnt descriptors, use the `ML/data_processing_training.py` script after changing the path variables and other hyperparameters as: 
```bash
python ML/data_processing_training.py
```

For training the Prediction model which predicts the Probability of Occurence at Equilibrium use the `ML/property_predicition.py` script. This requires the trained Encoder model weights, change the corresponding path of the encoder to the best performing encoder weights from the previous script.
```bash
python ./ML/property_predicition.py
```

For getting the metrics like RMSE, Residual on the whole dataset (Requires trained Encoder and Prediction model weights paths) use the `ML/value_check.py` by changing the path to the weights of the best performing Prediction model and corresponding encoder from the previous script.
```bash
python ./ML/value_check.py
```

If you have any suggestions/doubts feel free to raise an issue/PR on the repo or reach out to Ishan Kumar (ishankumar216@gmail.com).

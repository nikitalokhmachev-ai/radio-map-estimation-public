# Radio Map Estimation

This repository is the official implementation of all the models mentioned in the Radio Map Estimation with Deep Dual Path
Autoencoders and Skip Connection Learning paper.

## Project File Structure

- <b>data_utils.py</b> contains all the data-related utilities. Specifically, it defines the pytorch implementation of the radiomap dataset and the data scaler used in the paper.
- <b>models</b> contains all the models mentioned in the paper. The <b>autoencoder.py</b> file contains an abstract autoencoder class implementation. The classes in <b>autoencoders.py</b> inherit the hyperparameters and methods of the autoencoder class to define all the architectures of the paper. The rest of the files correspond to specific autoencoder arhitectures and contain implementations of encoders and decoders for each of them.
- <b>scalers</b> has all the pretrained scalers used in the paper. They are needed to reproduce the experiments.
- <b>notebooks</b> contains all the notebooks required to reproduce the results of the paper.
- <b>dataset</b> contains all the files needed to generate the dataset.

## Data Format

WIP

## Reproducibility

In order to reproduce the paper results, follow the following procedure:

1. Use the <b>Dataset Generation.ipynb</b> notebook to generate your own radiomap dataset.
2. In order to train a model, run the <b>ARL_Training_Example.ipynb</b> notebook. Specify what model you would like to train in the imports and model definition.
3. After training, execute the <b>Generate Predictions Based on Binned Data.ipynb</b> notebook. Before execution, specify the path to the model.
4. In order to visualize results, run the <b>Visualize Results.ipynb</b> notebook.

The paths in every notebook have to be changed according to your file structure.

## References

The code to generate the dataset as well as the dataset itself have been taken from [deep-autoencoders-cartography](https://github.com/fachu000/deep-autoencoders-cartography).

## Warining

Due to the stochastic nature of the data generation process as well as the difference in GPU architectures, your results might vary slightly.

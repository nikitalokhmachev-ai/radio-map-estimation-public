# Radio Map Estimation

This repository is the official implementation of all the models mentioned in the Radio Map Estimation with Deep Dual Path Autoencoders and Skip Connection Learning paper.

## Project File Structure

- <b>data_utils.py</b> contains all the data-related utilities. Specifically, it defines the pytorch implementation of the radiomap dataset and the data scaler used in the paper.
- <b>models</b> contains all the models mentioned in the paper. The <b>autoencoder.py</b> file contains an abstract autoencoder class implementation. The classes in <b>autoencoders.py</b> inherit the hyperparameters and methods of the autoencoder class to define all the architectures of the paper. The rest of the files correspond to specific autoencoder arhitectures and contain implementations of encoders and decoders for each of them.
- <b>scalers</b> has all the pretrained scalers used in the paper. They are needed to reproduce the experiments.
- <b>notebooks</b> contains all the notebooks required to reproduce the results of the paper.
- <b>dataset</b> contains all the files needed to generate a new dataset.

## Reproducibility

In order to reproduce the paper results, follow the following procedure:

1. Clone the current repository.

2. Make sure you have all the dependencies listed in requirements.txt. A simple way to do this is to create a virtual environment, e.g. using venv (https://docs.python.org/3/library/venv.html), and download the required dependencies there:
```
python -m venv radio_map_estimation
source radio_map_estimation/bin/activate
pip install -r respository/requirements.txt
```
Where `radio_map_estimation` and `respository` stand in for the full paths to where you want to save the new virtual environment and where you saved the cloned repository respectively.

3. Download the [Train](https://drive.google.com/file/d/1-z1gWOLLjD9O0K0whbCA7DsUJt64x6iq/view?usp=sharing), [Validation](https://drive.google.com/file/d/1-ONtHgLgNkI-kPAkdsta0DVkPfjS73js/view?usp=sharing), and [Test](https://drive.google.com/file/d/1KjCLM6DFGDwiIk_DIr005NsEeTbgRoXn/view?usp=sharing) datasets. The downloaded files are tarred and zipped and take about 2 GB, 208 MB, and 215 MB respectively. Their unzipped contents are about 7.36 GB, 819 MB, and 819 MB respectively.

3. To train a model, run <b>Train_Model.ipynb</b> in <b>notebooks</b> and follow instructions there.

3. After training, run <b>Evaluate_Model.ipynb</b> in <b>notebooks</b>. Before execution, specify the path to the model. <b>(To Do)</b>

4. To visualize the results, run <b>Visualize Results.ipynb</b> in <b>notebooks</b>. <b>(To Do)</b>

The paths in every notebook have to be changed according to your file structure.

## References

The code to generate the dataset as well as the dataset itself have been taken from [deep-autoencoders-cartography](https://github.com/fachu000/deep-autoencoders-cartography).

## Warning

Due to the stochastic nature of the data generation process as well as the difference in GPU architectures, your results might vary slightly.

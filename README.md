# Installing
Activate virtual environment:

```
python -m venv venv
source venv/bin/activate
```

Install libraries:

```
pip install -r requirements.txt
```

# Getting the datasets
 - https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz
 - https://www.kaggle.com/sripaadsrinivasan/fish-species-image-data
 - https://www.kaggle.com/ryanpartridge01/wild-edible-plants
 - https://www.kaggle.com/gpiosenka/100-bird-species
 - Cifar is downloaded through the script

# Running
To run training on all datasets and color models run the `main.py` script, or to run only one dataset use the corresponding script named after the daset, e.g. `fish.py`.

`main.py` does 5 runs of 25 epochs for each combination, while the individual scripts are conmfigured to do one run of 10 epochs for each color model.

# Utility scripts
We have also included a few utility scripts used for analysing the datasets and our results. These are `dataset_averages.py`, `averages.py` and `plotter.py`.

`averages.py` combines the csv files from multiple runs into a average csv, and `plotter.py` creates png plots from csv files.

`dataset_averages.py` is used to compute some statistics of the datasets, namely the mean, average and variance of each color channel (for all color models).

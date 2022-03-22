# NUS CS5340 Uncertainty Modelling in AI, Project

## Setup

It is recommended to do the following in a virtual environment.

### OpenTraj

We use `OpenTraj` to handle the data. To install, first update the git submodule

``` sh
git submodule update --init --recursive
```

Then install by `pip install -e OpenTraj`

### Project code

The folder `uncertainty-motion-prediction` contains a python package of the
project code. It includes our implementation of different methods.

Install by `pip install -e uncertainty-motion-prediction`.

## Workflow

Let's do the following:

- Implement the prediction algorithms. See `uncertainty-motion-prediction/src/uncertainty_motion_prediction/predictor/abstract.py`.
- Evaluate each of them in separate Jupyter notebooks. See `notebooks/constant-vel.ipynb`

For the common code (such as evaluation and data handling), implement in `uncertainty-motion-prediction`.

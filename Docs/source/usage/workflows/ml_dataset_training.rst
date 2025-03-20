.. _ml_dataset_training:

Training a Surrogate Model from WarpX Data
==========================================

Suppose we have a WarpX simulation that we wish to replace with a neural network surrogate model.
For example, a simulation determined by the following input script

.. dropdown:: Python Input for Training Simulation
   :color: light
   :icon: info
   :animate: fade-in-slide-down

    .. literalinclude:: ml_materials/run_warpx_training.py
       :language: python

In this section we walk through a workflow for data processing and model training, using data from this input script as an example.
The simulation output is stored in an online `Zenodo archive <https://zenodo.org/records/10368972>`__, in the ``lab_particle_diags`` directory.
In the example scripts provided here, the data is downloaded from the Zenodo archive, properly formatted, and used to train a neural network.
This workflow was developed and first presented in :cite:t:`ml-SandbergIPAC23,ml-SandbergPASC24`.
It assumes you have an up-to-date environment with PyTorch and openPMD.

Data Cleaning
-------------

It is important to inspect the data for artifacts, to
check that input/output data make sense.
If we plot the final phase space of the particle beam,
shown in :numref:`fig_unclean_phase_space`.
we see outlying particles.
Looking closer at the z-pz space, we see that some particles were not trapped in the accelerating region of the wake and have much less energy than the rest of the beam.

.. _fig_unclean_phase_space:

.. figure:: https://gist.githubusercontent.com/RTSandberg/649a81cc0e7926684f103729483eff90/raw/095ac2daccbcf197fa4e18a8f8505711b27e807a/unclean_stage_0.png
   :alt: Plot showing the final phase space projections of a particle beam through a laser-plasma acceleration element where some beam particles were not accelerated.

   The final phase space projections of a particle beam through a laser-plasma acceleration element where some beam particles were not accelerated.

To assist our neural network in learning dynamics of interest, we filter out these particles.
It is sufficient for our purposes to select particles that are not too far back, setting
``particle_selection={'z':[0.280025, None]}``.
After filtering, we can see in :numref:`fig_clean_phase_space` that the beam phase space projections are much cleaner -- this is the beam we want to train on.

.. _fig_clean_phase_space:

.. figure:: https://gist.githubusercontent.com/RTSandberg/649a81cc0e7926684f103729483eff90/raw/095ac2daccbcf197fa4e18a8f8505711b27e807a/clean_stage_0.png
   :alt: Plot showing the final phase space projections of a particle beam through a laser-plasma acceleration element after filtering out outlying particles.

   The final phase space projections of a particle beam through a laser-plasma acceleration element after filtering out outlying particles.

A particle tracker is set up to make sure
we consistently filter out these particles from both the initial and final data.

.. literalinclude:: ml_materials/create_dataset.py
   :language: python
   :dedent: 4
   :start-after: # Manual: Particle tracking START
   :end-before: # Manual: Particle tracking END

This data cleaning ensures that the particle data is distributed in a single blob,
as is optimal for training neural networks.

Create Normalized Dataset
-------------------------

Having chosen training data we are content with, we now need to format the data,
normalize it, and store the normalized data as well as the normalizations.
The script below will take the openPMD data we have selected and
format, normalize, and store it.

.. dropdown:: Python dataset creation
   :color: light
   :icon: info
   :animate: fade-in-slide-down

    .. literalinclude:: ml_materials/create_dataset.py
       :language: python

Load openPMD Data
^^^^^^^^^^^^^^^^^

First the openPMD data is loaded, using the particle selector as chosen above.
The neural network will make predictions from the initial phase space coordinates,
using the final phase space coordinates to measure how well it is making predictions.
Hence we load two sets of particle data, the source and target particle arrays.

.. literalinclude:: ml_materials/create_dataset.py
   :language: python
   :dedent: 4
   :start-after: # Manual: Load openPMD START
   :end-before: # Manual: Load openPMD END

Normalize Data
^^^^^^^^^^^^^^

Neural networks learn better on appropriately normalized data.
Here we subtract out the mean in each coordinate direction and
divide by the standard deviation in each coordinate direction,
for normalized data that is centered on the origin with unit variance.

.. literalinclude:: ml_materials/create_dataset.py
   :language: python
   :dedent: 4
   :start-after: # Manual: Normalization START
   :end-before: # Manual: Normalization END

openPMD to PyTorch Data
^^^^^^^^^^^^^^^^^^^^^^^

With the data normalized, it must be stored in a form PyTorch recognizes.
The openPMD data are 6 lists of arrays, for each of the 6 phase space coordinates
:math:`x, y, z, p_x, p_y,` and :math:`p_z`.
This data are converted to an :math:`N\times 6` numpy array and then to a PyTorch :math:`N\times 6` tensor.

.. literalinclude:: ml_materials/create_dataset.py
   :language: python
   :dedent: 4
   :start-after: # Manual: Format data START
   :end-before: # Manual: Format data END

Save Normalizations and Normalized Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The data is split into training and testing subsets.
We take most of the data (70%) for training, meaning that data is used to update
the neural network parameters.
The testing data is reserved to determine how well the neural network generalizes;
that is, how well the neural network performs on data that wasn't used to update the neural network parameters.
With the data split and properly normalized, it and the normalizations are saved to file for
use in training and inference.

.. literalinclude:: ml_materials/create_dataset.py
   :language: python
   :dedent: 4
   :start-after: # Manual: Save dataset START
   :end-before: # Manual: Save dataset END

Neural Network Structure
------------------------

It was found in :cite:t:`ml-SandbergPASC24` that a reasonable surrogate model is obtained with
shallow feedforward neural networks consisting of about 5 hidden layers and 700-900 nodes per layer.
The example shown here uses 3 hidden layers and 20 nodes per layer
and is trained for 10 epochs.

Some utility functions for creating neural networks are provided in the script below.
These are mostly convenience wrappers and utilities for working with `PyTorch <https://pytorch.org/>`__ neural network objects.
This script is imported in the training scripts shown later.

.. dropdown:: Python neural network class definitions
   :color: light
   :icon: info
   :animate: fade-in-slide-down

    .. literalinclude:: ml_materials/neural_network_classes.py
       :language: python3

Train and Save Neural Network
-----------------------------

The script below trains the neural network on the dataset just created.
In subsequent sections we discuss the various parts of the training process.

.. dropdown:: Python neural network training
   :color: light
   :icon: info
   :animate: fade-in-slide-down

    .. literalinclude:: ml_materials/train.py
       :language: python3

Training Function
^^^^^^^^^^^^^^^^^

In the training function, the model weights are updated.
Iterating through batches, the loss function is evaluated on each batch.
PyTorch provides automatic differentiation, so the direction of steepest descent
is determined when the loss function is evaluated and the ``loss.backward()`` function
is invoked.
The optimizer uses this information to update the weights in the ``optimizer.step()`` call.
The training loop then resets the optimizer and updates the summed error for the whole dataset
with the error on the batch and continues iterating through batches.
Note that this function returns the sum of all errors across the entire dataset,
which is later divided by the size of the dataset in the training loop.

.. literalinclude:: ml_materials/train.py
   :language: python
   :start-after: # Manual: Train function START
   :end-before: # Manual: Train function END

Testing Function
^^^^^^^^^^^^^^^^

The testing function just evaluates the neural network on the testing data that has not been used
to update the model parameters.
This testing function requires that the testing dataset is small enough to be loaded all at once.
The PyTorch dataloader can load data in batches if this size assumption is not satisfied.
The error, measured by the loss function, is returned by the testing function to be aggregated and stored.
Note that this function returns the sum of all errors across the entire dataset,
which is later divided by the size of the dataset in the training loop.

.. literalinclude:: ml_materials/train.py
   :language: python
   :start-after: # Manual: Test function START
   :end-before: # Manual: Test function END

Training Loop
^^^^^^^^^^^^^

The full training loop performs ``n_epochs`` number of iterations.
At each iteration the training and testing functions are called,
the respective errors are divided by the size of the dataset and recorded,
and a status update is printed to the console.

.. literalinclude:: ml_materials/train.py
   :language: python
   :start-after: # Manual: Training loop START
   :end-before: # Manual: Training loop END

Save Neural Network Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The model weights are saved after training to record the updates to the model parameters.
Additionally, we save some model metainformation with the model for convenience,
including the model hyperparameters, the training and testing losses, and how long the training took.

.. literalinclude:: ml_materials/train.py
   :language: python
   :start-after: # Manual: Save model START
   :end-before: # Manual: Save model END

Evaluate
--------

In this section we show two ways to diagnose how well the neural network is learning the data.
First we consider the train-test loss curves, shown in :numref:`fig_train_test_loss`.
This figure shows the model error on the training data (in blue) and testing data (in green) as a function of the number of epochs seen.
The training data is used to update the model parameters, so training error should be lower than testing error.
A key feature to look for in the train-test loss curve is the inflection point in the test loss trend.
The testing data is set aside as a sample of data the neural network hasn't seen before.
The testing error serves as a metric of model generalizability, indicating how well the model performs
on data it hasn't seen yet.
When the test-loss starts to trend flat or even upward, the neural network is no longer improving its ability to generalize to new data.

.. _fig_train_test_loss:

.. figure:: https://gist.githubusercontent.com/RTSandberg/649a81cc0e7926684f103729483eff90/raw/095ac2daccbcf197fa4e18a8f8505711b27e807a/beam_stage_0_training_testing_error.png
   :alt: Plot of training and testing loss curves versus number of training epochs.

   Training (in blue) and testing (in green) loss curves versus number of training epochs.

.. _fig_train_evaluation:

.. figure:: https://gist.githubusercontent.com/RTSandberg/649a81cc0e7926684f103729483eff90/raw/095ac2daccbcf197fa4e18a8f8505711b27e807a/beam_stage_0_model_evaluation.png
   :alt: Plot comparing model prediction with simulation output.

   A comparison of model prediction (yellow-red dots, colored by mean-squared error) with simulation output (black dots).

A visual inspection of the model prediction can be seen in :numref:`fig_train_evaluation`.
This plot compares the model prediction, with dots colored by mean-square error, on the testing data with the actual simulation output in black.
The model obtained with the hyperparameters chosen here trains quickly but is not very accurate.
A more accurate model is obtained with 5 hidden layers and 900 nodes per layer,
as discussed in :cite:t:`ml-SandbergPASC24`.

These figures can be generated with the following Python script.

.. dropdown:: Python visualization of progress training neural network
   :color: light
   :icon: info
   :animate: fade-in-slide-down

    .. literalinclude:: ml_materials/visualize.py
       :language: python3


Surrogate Usage in Accelerator Physics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A neural network such as the one we trained here can be incorporated in other BLAST codes.
Consider this `example using neural network surrogates of WarpX simulations in ImpactX <https://impactx.readthedocs.io/en/latest/usage/examples/pytorch_surrogate_model/README.html>`__.

.. bibliography::
   :keyprefix: ml-

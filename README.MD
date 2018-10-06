Student Symposium Poster for Model GUI Project:

[Link](GUI_poster.pdf)

This work performed under the auspices of the U.S. Department of Energy by Lawrence Livermore National Laboratory under Contract DE-AC52-07NA27344.

# MODEL_WRAPPER_GUI.PY #

> Allows Inception, ResNet, VGG, and LeNet models to be used with custom datasets without the need for using custom scripts tailored to each model. Helps non-experts with learning how to train and classify using existing pre-trained models.

# IMPORTANT NOTE #
As of the current TensorFlow version, r1.11, the LeNet model is no longer functional, as are the VGG Models. Currently working on a fix for these two models as well as implementing new types of classification (NLP, etc).

# HOW TO RUN #

- Install tensorflow, matplotlib, numpy, and pygubu
- If .tfrecords files (train-00000-of-00001 and validation-00000-of-00001) have not been generated, use:
     > ./run_build_image_data.sh
- Open GUI using:
     > python model_wrapper_gui.py

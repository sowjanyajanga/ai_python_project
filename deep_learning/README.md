Example handdigit_classification_with_tf.py
provided for Tendorflow in Chapter 20 works only on tensorflow 1.13.1 that needs Pythong 2.7
So, worked on a different conda environment for this module

Run TensorBoard from a terminal using the same Python 2.7 conda environment as the programs
tensorboard --logdir=graphs --port=8000

TO enable eager execution of tensors
tf.enable_eager_execution()


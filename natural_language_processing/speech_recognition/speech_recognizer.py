import hmmlearn as hmm
import numpy as np
# Class to handle all HMM related processing
class HMMTrainer(object):
    def __init__(self, model_name='GaussianHMM', n_components=4, cov_type='diag', n_iter=1000):
        self.model_name = model_name
        self.n_components = n_components
        self.cov_type = cov_type
        self.n_iter = n_iter
        self.models = []

        if self.model_name == 'GaussianHMM':
            self.model = hmm.GaussianHMM(n_components=self.n_components,
                                         covariance_type=self.cov_type, n_iter=self.n_iter)
        else:
            raise TypeError('Invalid model type')

    # X is a 2D numpy array where each row is 13D
    def train(self, train_data):
        np.seterr(all='ignore')
        self.models.append(self.model.fit(train_data))

    # Run the model on input data
    def get_score(self, input_data):
        return self.model.score(input_data)


import os
import argparse
import numpy as np
from scipy.io import wavfile
from hmmlearn import hmm
from python_speech_features import mfcc

# Function to parse input arguments
def build_arg_parser():
    parser = argparse.ArgumentParser(description='Trains the HMM classifier')
    parser.add_argument("--input-folder", dest="input_folder", required=True, help="Input folder containing the audio files in subfolders")
    return parser

if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    input_folder = args.input_folder
    hmm_models = []
    test_files = []
    # Parse the input directory
    for dirname in os.listdir(input_folder):
        # Get the name of the subfolder
        subfolder = os.path.join(input_folder, dirname)
        if not os.path.isdir(subfolder):
            continue
        # Extract the label
        label = subfolder[subfolder.rfind('/') + 1:]
        # Initialize variables
        X = np.array([])
        y_words = []
        # Iterate through the audio files (leaving 1 file for testing in each class )
        # for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav') ][:-1]:
        test_files.append([os.path.join(subfolder, x) for x in os.listdir(subfolder) if x.endswith('15.wav')][0])
        for filename in [x for x in os.listdir(subfolder) if (x.endswith('.wav') and not x.endswith('15.wav'))]:
            # print(filename)
            # Read the input file
            filepath = os.path.join(subfolder, filename)
            sampling_freq,audio = wavfile.read(filepath)
            # Extract MFCC features
            mfcc_features = mfcc(audio, sampling_freq)
            # Append to the variable X
            if len(X) == 0:
                X = mfcc_features
            else:
                X = np.append(X, mfcc_features, axis=0)
            # Append the label
            y_words.append(label)

        # print(X)
        # print(y_words)
        # Train and save HMM model
        hmm_trainer = HMMTrainer()
        hmm_trainer.train(X)
        hmm_models.append((hmm_trainer, label))
        hmm_trainer = None

    # Test files
    # print(test_files)

    # Classify input data
    for test_file in test_files:
        # Read input file
        sampling_freq, audio = wavfile.read(test_file)
        # Extract MFCC features
        mfcc_features = mfcc(audio, sampling_freq)
        # Define variables
        max_score = float('-inf')
        output_label = None
        # Iterate through all HMM models and pick # the one with the highest score
        for item in hmm_models:
            hmm_model, label = item
            score = hmm_model.get_score(mfcc_features)
            if score > max_score:
                max_score = score
                output_label = label

        # Print the output
        print("True:",test_file[test_file.find(' / ')+1: test_file.rfind(' / ')])
        print("Predicted:", output_label)




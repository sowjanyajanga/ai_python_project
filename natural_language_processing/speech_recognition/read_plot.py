import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os

# Read the input file
wave_file_loc = os.path.dirname(os.path.abspath(__name__)) + "/natural_language_processing/Dataset/input_read.wav"
sampling_freq, audio = wavfile.read(wave_file_loc)

# Print the params
print('Shape:', audio.shape)
print('Datatype:', audio.dtype)
print('Frequency:', sampling_freq)
print('Duration:', round(audio.shape[0] / float(sampling_freq),3), 'seconds')

# Normalize the values
audio = audio / (2.**15)

# Extract first 30 values for plotting
audio = audio[:30]

# Build the time axis
x_values = np.arange(0, len(audio), 1) / float(sampling_freq)

# Convert to seconds
x_values *= 1000

# Plotting the chopped audio signal
plt.plot(x_values, audio, color='black')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.title('Audio signal')
plt.show()


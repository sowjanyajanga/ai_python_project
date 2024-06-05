import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import os

# Read the input file
wave_file_loc = os.path.dirname(os.path.abspath(__name__)) + "/natural_language_processing/Dataset/input_freq.wav"
sampling_freq, audio = wavfile.read(wave_file_loc)

# Normalize the values
audio = audio / (2.**15)

# Get length
len_audio = len(audio)

# Apply Fourier transform
transformed_signal = np.fft.fft(audio)
half_length = np.ceil((len_audio + 1) / 2.0)

transformed_signal = abs(transformed_signal[0:int(half_length)])
transformed_signal /= float(len_audio)
transformed_signal **= 2

# Extract length of transformed signal
len_ts = len(transformed_signal)

# Extract power in dB
power = 10 * np.log10(transformed_signal)

# Build the time axis
x_values = np.arange(0, half_length, 1) * (sampling_freq /len_audio) / 1000.0

# Plot the figure
plt.figure()
plt.plot(x_values, power, color='black')
plt.xlabel('Freq (in kHz)')
plt.ylabel('Power (in dB)')
plt.show()



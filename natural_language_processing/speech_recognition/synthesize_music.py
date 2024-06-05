import json
import numpy as np
from scipy.io.wavfile import write
import os

# Synthesize tone
def synthesizer(freq, duration, amp=1.0, sampling_freq=44100):
    # Build the time axis
    t = np.linspace(0, duration, round(duration * sampling_freq))
    # Construct the audio signal
    audio = amp * np.sin(2 * np.pi * freq * t)
    return audio.astype(np.int16)

if __name__ == '__main__':
    tone_map_file = os.path.dirname(os.path.abspath(__name__)) + "/natural_language_processing/Dataset/tone_freq_map.json"
    # Read the frequency map
    with open(tone_map_file, 'r') as f:
        tone_freq_map = json.loads(f.read())

    # Set input parameters to generate 'G' tone
    input_tone = 'G'
    duration = 2  # seconds
    amplitude = 10000
    sampling_freq = 44100  # Hz

    # # Generate the tone
    # synthesized_tone = synthesizer(tone_freq_map[input_tone], duration, amplitude, sampling_freq)
    #
    # # Write to the output file
    # output_file = os.path.dirname(os.path.abspath(__name__)) + "/natural_language_processing/speech_recognition/output_tone.wav"
    # write(output_file, sampling_freq, synthesized_tone)

    tone_seq = [('D', 0.3), ('G', 0.6), ('C', 0.5), ('A', 0.3),('Asharp', 0.7)]
    output_file = os.path.dirname(os.path.abspath(__name__)) + "/natural_language_processing/speech_recognition/output_tone_seq.wav"

    # Construct the audio signal based on the chord sequence
    output = np.array([])
    for item in tone_seq:
        input_tone = item[0]
        duration = item[1]
        synthesized_tone = synthesizer(tone_freq_map[input_tone], duration, amplitude, sampling_freq)
        print(synthesized_tone)
        output = np.append(output, synthesized_tone, axis=0)
        output = output.astype(np.int16)

    # Write to the output file
    write(output_file, sampling_freq, output)
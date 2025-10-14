import os
import numpy as np
from scipy.io import wavfile
from scipy.fftpack import dct

def extract_mfcc(filepath, num_ceps=13, n_filters=26, nfft=512, sample_rate=16000):
    # Read audio
    sr, signal = wavfile.read(filepath)
    signal = signal.astype(float)

    # If it's stereo, choose mono.
    if signal.ndim == 2:
        signal = signal[:, 0]

    # Pre-emphasis
    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

    # Frame splitting
    frame_size = 0.025  # 25ms
    frame_stride = 0.01 # 10ms
    frame_length = int(round(frame_size * sr))
    frame_step = int(round(frame_stride * sr))
    signal_length = len(emphasized_signal)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z)

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # Windowing (Hamming Window)
    frames *= np.hamming(frame_length)

    # FFT and Power Spectrum
    mag_frames = np.absolute(np.fft.rfft(frames, nfft))
    pow_frames = ((1.0 / nfft) * (mag_frames ** 2))

    # Mel Filter bank
    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_filters + 2)
    hz_points = 700 * (10**(mel_points / 2595) - 1)
    bin = np.floor((nfft + 1) * hz_points / sample_rate)

    fbank = np.zeros((n_filters, int(np.floor(nfft / 2 + 1))))
    for m in range(1, n_filters + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # mid
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Avoid log(0)
    filter_banks = 20 * np.log10(filter_banks)  # Change to dB

    # DCT get MFCC
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :num_ceps]

    # Subtract the mean (to enhance robustness)
    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)

    return mfcc

# Batch process all audio files
data_dir = "speech_data/"     # Input voice folder
save_dir = "features/"        # Output feature folder
os.makedirs(save_dir, exist_ok=True)

features = {}

for file_name in os.listdir(data_dir):
    if file_name.lower().endswith(".wav"):
        file_path = os.path.join(data_dir, file_name)

        feats = extract_mfcc(file_path)

        np.save(os.path.join(save_dir, file_name.replace(".wav", ".npy")), feats)

        features[file_name] = feats

        print(f"{file_name} extraction complete，Shape: {feats.shape}")

print("\n All audio features have been extracted and saved to:", save_dir)
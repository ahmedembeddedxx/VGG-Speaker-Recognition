# Third Party
import librosa
import numpy as np
import soundfile as sf
import scipy.signal as sps

# ===============================================
#       Code for loading data.
# ===============================================
def load_wav(vid_path, sr, mode='train', use_fast=False):
    """Load audio file and extend it. Optionally reverse audio for data augmentation."""
    if use_fast:
        try:
            wav, sr_ret = sf.read(vid_path)
        except Exception as e:
            raise RuntimeError(f"Error reading {vid_path}: {e}")
    else:
        wav, sr_ret = librosa.load(vid_path, sr=None)
        if sr_ret != sr:
            raise ValueError(f"Expected sampling rate {sr}, but got {sr_ret}")

    if mode == 'train':
        extended_wav = np.append(wav, wav)
        if np.random.random() < 0.3:
            extended_wav = extended_wav[::-1]
    else:
        extended_wav = np.append(wav, wav[::-1])

    return extended_wav

def lin_spectrogram_from_wav(wav, hop_length, win_length, n_fft=1024):
    """Generate a linear spectrogram from the waveform."""
    return librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length).T

def load_data(path, win_length=400, sr=16000, hop_length=160, n_fft=512, spec_len=250, mode='train'):
    """Load audio data, compute its spectrogram, and preprocess it."""
    wav = load_wav(path, sr=sr, mode=mode, use_fast=True)
    linear_spect = lin_spectrogram_from_wav(wav, hop_length, win_length, n_fft)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    freq, time = mag_T.shape

    if mode == 'train':
        if time > spec_len:
            randtime = np.random.randint(0, time-spec_len)
            spec_mag = mag_T[:, randtime:randtime+spec_len]
        else:
            spec_mag = np.pad(mag_T, ((0, 0), (0, spec_len - time)), 'constant')
    else:
        spec_mag = mag_T

    # Preprocessing: subtract mean and divide by time-wise variance
    mu = np.mean(spec_mag, axis=0, keepdims=True)
    std = np.std(spec_mag, axis=0, keepdims=True)
    
    return (spec_mag - mu) / (std + 1e-5)

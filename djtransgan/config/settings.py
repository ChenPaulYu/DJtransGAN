# Sampling Rate
SR = 44100


# STFT Parameters
WINDOW      = 'hann'
N_FFT       = 2048  # 256 lose time resolution
HOP_LENGTH  = 512   # 128
N_MELS      = 128

# Band Parameters
BAND_FREQS = [20, 300, 5000, 20000]


# Avoid Divide Zeros
EPSILON = 1e-12
import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from scipy.fftpack import fft, fftfreq
import numpy as np

#Read the .wav file here:  ˇˇˇˇˇˇˇˇˇˇˇ
rate, data = wav.read(
    'wine_resonance.wav')


#Take the Fourier transform of the .wav file. Take the log10 of it and normalize (so Fourier_ynorm is between 0 and 1)
Fourier_y = np.log(abs(fft(data[:, 0])))
Fourier_ynorm = Fourier_y/max(Fourier_y)
Fourier_x = fftfreq(len(data), 1/rate)
freq = np.fft.fftshift(Fourier_x)
fft_data = np.fft.fftshift(Fourier_ynorm)
#The x-axis is in Hz, so a 1200 Hz beep would give a peak at 1200 and -1200

#These two lines plot the Fourier transform, but there is a weird horizontal line at y = 0.3
plt.plot(Fourier_x, Fourier_ynorm, '.')
plt.xlim(1200, 1400)
plt.show()
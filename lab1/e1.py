import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import sounddevice as sd
import soundfile as sf

data, fs = sf.read('sound1.wav', dtype='float32')

print(data.dtype)
print(data.shape)

# sd.play(data, fs)
# status = sd.wait()

sf.write('sound_L.wav', data[0], fs)
sf.write('sound_R.wav', data[1], fs)
sf.write('sound_mix.wav', (data[1]+data[0])/2, fs)

# Signal
plt.figure()
plt.subplot(2,1,1)
plt.plot(data[:,0])
plt.subplot(2,1,2)
plt.plot(data[:,1])
plt.show()

# Widmo
plt.figure()
plt.subplot(2,1,1)
plt.plot(np.arange(0,data[:,0].shape[0])/fs,data[:,0])

plt.subplot(2,1,2)
yf = scipy.fftpack.fft(data[:,0])
plt.plot(np.arange(0,fs,1.0*fs/(yf.size)),np.abs(yf))
plt.show()

fsize=2**8

plt.figure()
plt.subplot(2,1,1)
plt.plot(np.arange(0,data[:,0].shape[0])/fs,data[:,0])

plt.subplot(2,1,2)
yf = scipy.fftpack.fft(data[:,0],fsize)

plt.plot(np.arange(0,fs,fs/fsize),np.abs(yf))
plt.show()

plt.figure()
plt.subplot(2,1,1)
plt.plot(np.arange(0,data[:,0].shape[0])/fs,data[:,0])

plt.subplot(2,1,2)
yf = scipy.fftpack.fft(data[:,0],fsize)

plt.plot(np.arange(0,fs/2,fs/fsize),np.abs(yf[:fsize//2]))
plt.show()

plt.figure()
plt.subplot(2,1,1)
plt.plot(np.arange(0,data[:,0].shape[0])/fs,data[:,0])

plt.subplot(2,1,2)
yf = scipy.fftpack.fft(data[:,0],fsize)

plt.plot(np.arange(0,fs/2,fs/fsize),20*np.log10( np.abs(yf[:fsize//2])))
plt.show()

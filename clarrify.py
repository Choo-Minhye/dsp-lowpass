from scipy.io.wavfile import read
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
import os
import librosa

# t1 = t1 * 1000 #Works in milliseconds
# t2 = t2 * 1000
# newAudio = AudioSegment.from_wav("oldSong.wav")
# newAudio = newAudio[t1:t2]
# newAudio.export('newSong.wav', format="wav") 

def calc_distances(sound_file):
    #The minimun value for the sound to be recognized as a knock
    min_val = 4000
    
    fs, data = read(sound_file)
    data_size = len(data)
    
    print("Data: ", data, data.dtype)
# print("hello for debug?")
    print("Range of amplitude: ", min(data), max(data))
    print("Mean of amplitude: ", np.mean(data))
    print("Standard Deviation of amplitude: ", np.std(data)) # 표준편차
    plt.plot(data)
    plt.title("Time domain")
    plt.ylabel("Amplitude")
    plt.xlabel("Time (samples)")
    # plt.savefig("original_time_domain.png", bbox_inches='tight')
    plt.show()
    
    
    
    #The number of indexes on 0.15 seconds
    focus_size = int(0.15 * fs)
    
    focuses = []
    distances = []
    idx = 0
    
    while idx < len(data):
        if data[idx] > min_val:
            mean_idx = idx + focus_size // 2
            # print(mean_idx)
            focuses.append(float(mean_idx) / data_size)
            if len(focuses) > 1:
                last_focus = focuses[-2]
                actual_focus = focuses[-1]
                distances.append(actual_focus - last_focus)
            idx += focus_size
        else:
            idx += 1

    return distances  
calc_distances('denoise.wav')



def trim_audio_data(audio_file, save_file):
    sr = 96000
    sec = 3

    y, sr = librosa.load(audio_file, sr=sr)

    ny = y[:sr*sec]

    librosa.output.write_wav(save_file + '.wav', ny, sr)

# base_path = 'dataset/'

audio_path = 'dsp-lowpass/audio'
save_path = 'dsp-lowpass/save'


audio_file = audio_path + '/' + 'denoise.wav'
save_file = 'split_denoise.wav'[:-4]

trim_audio_data('denoise.wav', save_file)


# print(calc_distances('denoise.wav'))


# import numpy as np
# from matplotlib import pyplot as plt
# import numpy as np
# from scipy.fftpack import fft, ifft
# from scipy.io import wavfile
# import sounddevice as sd

# # OPEN THE SOUND FILE
# print("Opening the sound file... \n")
# # sample_rate, data = wavfile.read('noisy_sound.wav')  # load the data
# # sample_rate, data = wavfile.read('183_1b1_Tc_sc_Meditron_16bit.wav')  # load the data
# # sample_rate, data = wavfile.read('DP78_Asthma,E W,P R M,20,M.wav')  # load the data
# sample_rate, data = wavfile.read('DP46_asthma,E W,P L U,41,F.wav')  # load the data
# samples = len(data)
# print("Total samples = ", samples)
# sd.play(data, sample_rate)

# print("Data: ", data, data.dtype)
# # print("hello for debug?")
# print("Range of amplitude: ", min(data), max(data))
# plt.plot(data)
# plt.title("Time domain")
# plt.ylabel("Amplitude")
# plt.xlabel("Time (samples)")
# # plt.savefig("original_time_domain.png", bbox_inches='tight')
# plt.show()

# print("mean = ",data.mean())

# # create noise data
# def function(x, noise):
#     y = np.sin(7*x+2) + noise
#     return y

# def function2(x, noise):
#     y = np.sin(6*x+2) + noise
#     return y


# noise = np.random.uniform(low=-0.3, high=0.3, size=(100,))
# x_line0 = np.linspace(1.95,2.85,100)
# y_line0 = function(x_line0, noise)
# x_line = np.linspace(0, 1.95, 100)
# x_line2 = np.linspace(2.85, 3.95, 100)
# x_pik = np.linspace(3.95, 5, 100)
# y_pik = function2(x_pik, noise)
# x_line3 = np.linspace(5, 6, 100)

# # concatenate noise data
# x = np.linspace(0, 6, 500)
# y = np.concatenate((noise, y_line0, noise, y_pik, noise), axis=0)

# # plot data
# noise_band = 1.1
# top_noise = y.mean()+noise_band*np.amax(noise)
# bottom_noise = y.mean()-noise_band*np.amax(noise)
# fig, ax = plt.subplots()
# ax.axhline(y=y.mean(), color='red', linestyle='--')
# ax.axhline(y=top_noise, linestyle='--', color='green')
# ax.axhline(y=bottom_noise, linestyle='--', color='green')
# ax.plot(x, y)

# # split data into 2 signals
# def split(arr, cond):
#   return [arr[cond], arr[~cond]]

# # find bottom noise data indexes
# botom_data_indexes = np.argwhere(y < bottom_noise)
# # split by visual x value
# splitted_bottom_data = split(botom_data_indexes, botom_data_indexes < np.argmax(x > 3))

# # find top noise data indexes
# top_data_indexes = np.argwhere(y > top_noise)
# # split by visual x value
# splitted_top_data = split(top_data_indexes, top_data_indexes < np.argmax(x > 3))

# # get first signal range
# first_signal_start = np.amin(splitted_bottom_data[0])
# first_signal_end = np.amax(splitted_top_data[0])

# # get x index of first signal
# x_first_signal = np.take(x, [first_signal_start, first_signal_end])
# ax.axvline(x=x_first_signal[0], color='orange')
# ax.axvline(x=x_first_signal[1], color='orange')

# # get second signal range
# second_signal_start = np.amin(splitted_top_data[1])
# second_signal_end = np.amax(splitted_bottom_data[1])

# # get x index of first signal
# x_second_signal = np.take(x, [second_signal_start, second_signal_end])
# ax.axvline(x=x_second_signal[0], color='orange')
# ax.axvline(x=x_second_signal[1], color='orange')

# plt.show()
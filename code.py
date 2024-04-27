####################### Encryption Process ################################
####################### 2D of our equations ################################
import numpy as np
import matplotlib.pyplot as plt
import librosa
import hashlib
import random
from PyEMD import EMD
from ast import literal_eval
from math import cos
import seaborn as sns
import time
from matplotlib.pyplot import savefig

#read audio file
with open("/content/CantinaBand3.wav",'rb') as read_audio:
    org_aud=read_audio.read()
# remove header
header=org_aud[0:44]
audio=org_aud[44:]
audio=list(audio)

def enc_proc(audio):

    #create hash value
    hash_val=hashlib.sha3_512((repr(audio).encode())).hexdigest()

    #Divide hash value into 8 equal component
    components=[int(hash_val[i:i+16], 16) for i in range(0, 128, 16)]

    #create 4 initial parameters for the modified equation
    k = [(components[i] ^ components[i+1]) / (2**68) + (2.9 if i in (4, 6) else 0) for i in range(0, 7, 2)]

    # Generate initial values for x and q
    x = components[0] / (2 ** 64)
    q = components[1] / (2 ** 64)

    #generate random number between 0, len(audio) using hash value1
    random.seed(hash_val)
    index=random.sample(range(0, (len(audio))), len(audio))

    #shuffle audio according to index1
    suff_aud=np.zeros(len(audio))
    suff_aud=np.array(audio)[index]

    # key generate using the modified equations
    def keygen(x,y,r1,r2,size):
        key1=[]
        key2=[]
        for i in range(size):
            x = k[0] * x * (q ** 2 - 1)
            y = y + x
            key1.append((int(x*pow(10,16))%256))
            key2.append((int(y*pow(10,16))%256))
        return key1,key2
    key1,key2=keygen(k[0],k[1],k[2],k[3],len(audio))
    key=(np.array(key1))^(np.array(key2))

    #determine the height and width of the 2d matrix
    factor = [i for i in range(1, len(suff_aud) + 1) if len(suff_aud) % i == 0]
    width,height = factor[len(factor) // 2 - 1], len(suff_aud) // factor[len(factor) // 2 - 1]

     #convert audio into 2d matrix
    td_mat=np.array(np.reshape(suff_aud,(height,width)),dtype='uint8')

    #calculate IMFs and resudual of the each row of the 2d matrix
    sum_IMF=[]
    all_residual=[]
    for i in range(height):
        signal=td_mat[i][:]
        emd = EMD(DTYPE=np.float32)
        energy=float('inf')
        IMFs=[]
        engy=[]
        while(energy):
            IMF = emd.emd(signal,max_imf=1)
            IMF=np.int32(IMF)
            IMFs.append(IMF[0])
            signal=signal-IMF[0]
            energy=((np.sum(signal**2))/len(signal))
            engy.append(energy)
            if (len(engy)>=2) and (engy[-1]==engy[-2]):
                break
        sum_IMF.append(np.sum(IMFs,axis=0))
        all_residual.append(signal)

    #Encryption process of resudual
    enc_residual=np.zeros(shape=[height,width],dtype='uint8')
    k=0
    for i in range(height):
        for j in range(width):
            enc_residual[i][j]=((all_residual[i][j])^key[k])
            k+=1

    return sum_IMF, enc_residual, index, key

sum_IMF, enc_residual, index, key = enc_proc(audio)

#encrypt_signal=sum_IMF+enc_residual
encrypt_signal=np.zeros(shape=[enc_residual.shape[0],enc_residual.shape[1]],dtype='uint8')
for i in range(enc_residual.shape[0]):
    encrypt_signal[i,:] = sum_IMF[i] + enc_residual[i,:]

#convert encrypted 2d matrix to 1D audio
encpt_aud = np.reshape(encrypt_signal, -1)

#convert to bytes
encpt_aud_bytes = bytes(encpt_aud)

#add header
encpt_main_audio = header + encpt_aud_bytes

#save encrypted audio
with open('/content/encryptedaudio.wav', 'wb') as writefile:
    encrpt_file_imf = writefile.write(encpt_main_audio)


############################### Decryption Process #######################

def decryption_proc(sum_IMF, enc_residual, index, key):
    height = enc_residual.shape[0]
    width = enc_residual.shape[1]

    #decryption of residual
    decryp_residual = np.zeros(shape=[height, width], dtype='uint8')
    k = 0
    for i in range(height):
        for j in range(width):
            decryp_residual[i][j] = (enc_residual[i][j]) ^ key[k]
            k += 1

    #decrypt_signal=sum_IMF+dec_residual
    decrypt_signal = np.zeros(shape=[height, width], dtype='uint8')
    for i in range(height):
        decrypt_signal[i,:] = sum_IMF[i] + decryp_residual[i,:]

    #convert decrypt_signal to 1d array
    decry_aud_1d = np.reshape(decrypt_signal, -1)

    #shuffle decry_aud
    shuffle_aud = np.zeros(height * width)
    shuffle_aud[index] = decry_aud_1d

    #convert into bytes
    shuffle_aud = shuffle_aud.astype('uint8')
    decrypted_audio = bytes(shuffle_aud)

    return decrypted_audio

decrypted_signal = decryption_proc(sum_IMF, enc_residual, index, key)

##add header to decrypted_signal
decrypted_audio = header + decrypted_signal

####save decrypted audio
with open('/content/decript2.wav', 'wb') as writefile:
    decrpt_file_imf = writefile.write(decrypted_audio)

#read all the audio
y, fs = librosa.load('/content/decript2.wav')
y1,fs1=librosa.load('/content/encryptedaudio.wav')
y2,fs2=librosa.load('/content/CantinaBand3.wav')

#plot all the audio
fig, axes = plt.subplots(1, 3, figsize=(20, 4), sharey=True)

sns.lineplot(y2,ax=axes[0])
axes[0].set(xlabel ="Time (s)", ylabel = "Amplitude",title="Original Audio")
sns.lineplot(y1,ax=axes[1])
axes[1].set(xlabel ="Time (s)", ylabel = "Amplitude",title="Encrypted Audio")
sns.lineplot(y,ax=axes[2])
axes[2].set(xlabel ="Time (s)", ylabel = "Amplitude",title="Decrypted Audio")

plt.show()
#savefig('../results/display_audio.png')
######################################## Entropy ################################

from skimage.measure.entropy import shannon_entropy
print('Entropy of Encrypted audio:',shannon_entropy(encrypt_signal[:,:]))

##################################### histogram ################

fig, axes = plt.subplots(1, 2, figsize=(11,4), sharey=False)

sns.histplot(y2,ax=axes[0])
axes[0].set(xlabel ="Amplitude", ylabel = "Number of samples",title="Histogram of Original Audio")
sns.histplot(y1,ax=axes[1])
axes[1].set(xlabel ="Amplitude", ylabel = "Number of samples",title="Histogram of Encrypted Audio")
plt.show()
#savefig('../results/histogram.png')
##############################correlation analysis###########################################

#horizontal

X1=y2[0:-2]
Y1=y2[1:-1]
rand_index1=np.random.permutation(len(X1))
rand_index1_H=rand_index1[0:50000]
x1_h=X1[rand_index1_H]
y1_h=Y1[rand_index1_H]
c1_horz=np.corrcoef(x1_h, y1_h)
print("correlation of Original Audio",c1_horz[0][1])

X=y1[0:-2]
Y=y1[1:-1]
rand_index=np.random.permutation(len(X))
rand_index_H=rand_index[0:50000]
x_h=X[rand_index_H]
y_h=Y[rand_index_H]
c_horz=np.corrcoef(x_h, y_h)
print("correlation of Original Audio",c_horz[0][1])


fig, axes = plt.subplots(1, 2, figsize=(11,4), sharey=False)


axes[0].scatter(x1_h.flat, y1_h.flat, marker='.',s=1)
axes[0].set(xlabel ="Amplitude", ylabel = "Number of samples",title="Correlation of Original Audio")
axes[1].scatter(x_h.flat, y_h.flat, marker='.',s=1)
axes[1].set(xlabel ="Amplitude", ylabel = "Number of samples",title="Correlation of Encrypted Audio")
plt.show()
#savefig('../results/correlation.png')
########################################Differential Attack Analysis ###################################

#Differential Attack Analysis
audio[60000]=56
sum_IMF,enc_residual,index,key=enc_proc(audio)
#encrypt_signal=sum_IMF+enc_residual
encrypt_signal=np.zeros(shape=[enc_residual.shape[0],enc_residual.shape[1]],dtype='uint8')
for i in range (enc_residual.shape[0]):
    encrypt_signal[i][:]=sum_IMF[i]+enc_residual[i]

#convert encrypted 2d matrix to 1D audio

encpt_aud1=np.reshape(encrypt_signal,-1)

#convert to bytes
encpt_aud_bytes=bytes(encpt_aud)

#add header
encpt_main_audio=header+encpt_aud_bytes

#save encrypted audio
with open('/content/encryptedaudio.wav','wb') as writefile:
        encrpt_file_imf=writefile.write(encpt_main_audio)

y3, fs3 = librosa.load('/content/encryptedaudio.wav')


#NSCR
D=[]
for i in range(len(y)):
    if y2[i]!=y3[i]:
        D.append(1)
    else:
        D.append(0)
NSCR=((np.sum(D))/len(y))*100
print("The value of NSCR is",NSCR)


#UACI
size=len(encpt_aud)
encrypted_aud_int=np.array(encpt_aud,dtype='int64')
encrypted_aud11_int=np.array(encpt_aud1,dtype='int64')
diff=(encrypted_aud_int-encrypted_aud11_int)
abs_diff=(np.absolute(diff))
uaci=(np.sum(abs_diff))/(size*255)*100
print("The value of UACI is",uaci)

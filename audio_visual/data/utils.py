import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from scipy import signal
from scipy.io import wavfile
import cv2 as cv
from scipy.special import softmax



def prepare_main_input(audioFile, visualFeaturesFile, targetFile, noise, reqInpLen, charToIx, noiseSNR, audioParams, videoParams):

    """
    Function to convert the data sample in the main dataset into appropriate tensors.
    """

    if targetFile is not None:

        with open(targetFile, "r") as f:
            trgt = f.readline().strip()[7:]

        trgt = [charToIx[char] for char in trgt]
        trgt.append(charToIx["<EOS>"])
        trgt = np.array(trgt)
        trgtLen = len(trgt)

        if trgtLen > 100:
            print("Target length more than 100 characters. Exiting")
            exit()


    stftWindow = audioParams["stftWindow"]
    stftWinLen = audioParams["stftWinLen"]
    stftOverlap = audioParams["stftOverlap"]
    sampFreq, inputAudio = wavfile.read(audioFile)

    if len(inputAudio) < sampFreq*(stftWinLen + 3*(stftWinLen - stftOverlap)):
        padding = int(np.ceil((sampFreq*(stftWinLen + 3*(stftWinLen - stftOverlap)) - len(inputAudio))/2))
        inputAudio = np.pad(inputAudio, padding, "constant")
    inputAudio = inputAudio/np.max(np.abs(inputAudio))

    if noise is not None:
        pos = np.random.randint(0, len(noise)-len(inputAudio)+1)
        noise = noise[pos:pos+len(inputAudio)]
        noise = noise/np.max(np.abs(noise))
        gain = 10**(noiseSNR/10)
        noise = noise*np.sqrt(np.sum(inputAudio**2)/(gain*np.sum(noise**2)))
        inputAudio = inputAudio + noise

    inputAudio = inputAudio/np.sqrt(np.sum(inputAudio**2)/len(inputAudio))

    _, _, stftVals = signal.stft(inputAudio, sampFreq, window=stftWindow, nperseg=sampFreq*stftWinLen, noverlap=sampFreq*stftOverlap,
                                 boundary=None, padded=False)
    audInp = np.abs(stftVals)
    audInp = audInp.T


    vidInp = np.load(visualFeaturesFile)


    if len(audInp)/4 >= len(vidInp):
        inpLen = int(np.ceil(len(audInp)/4))
        leftPadding = int(np.floor((4*inpLen - len(audInp))/2))
        rightPadding = int(np.ceil((4*inpLen - len(audInp))/2))
        audInp = np.pad(audInp, ((leftPadding,rightPadding),(0,0)), "constant")
        leftPadding = int(np.floor((inpLen - len(vidInp))/2))
        rightPadding = int(np.ceil((inpLen - len(vidInp))/2))
        vidInp = np.pad(vidInp, ((leftPadding,rightPadding),(0,0)), "constant")
    else:
        inpLen = len(vidInp)
        leftPadding = int(np.floor((4*inpLen - len(audInp))/2))
        rightPadding = int(np.ceil((4*inpLen - len(audInp))/2))
        audInp = np.pad(audInp, ((leftPadding,rightPadding),(0,0)), "constant")


    if inpLen < reqInpLen:
        leftPadding = int(np.floor((reqInpLen - inpLen)/2))
        rightPadding = int(np.ceil((reqInpLen - inpLen)/2))
        audInp = np.pad(audInp, ((4*leftPadding,4*rightPadding),(0,0)), "constant")
        vidInp = np.pad(vidInp, ((leftPadding,rightPadding),(0,0)), "constant")

    inpLen = len(vidInp)


    audInp = torch.from_numpy(audInp)
    vidInp = torch.from_numpy(vidInp)
    inp = (audInp,vidInp)
    inpLen = torch.tensor(inpLen)
    if targetFile is not None:
        trgt = torch.from_numpy(trgt)
        trgtLen = torch.tensor(trgtLen)
    else:
        trgt, trgtLen = None, None

    return inp, trgt, inpLen, trgtLen



def prepare_pretrain_input(audioFile, visualFeaturesFile, targetFile, noise, numWords, charToIx, noiseSNR, audioParams, videoParams):

    """
    Function to convert the data sample in the pretrain dataset into appropriate tensors.
    """

    with open(targetFile, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]

    trgt = lines[0][7:]
    words = trgt.split(" ")

    if len(words) <= numWords:
        trgtNWord = trgt
        sampFreq, inputAudio = wavfile.read(audioFile)
        vidInp = np.load(visualFeaturesFile)

    else:
        nWords = [" ".join(words[i:i+numWords]) for i in range(len(words)-numWords+1)]
        nWordLens = np.array([len(nWord)+1 for nWord in nWords]).astype(np.float)

        ix = np.random.choice(np.arange(len(nWordLens)), p=softmax(nWordLens))
        trgtNWord = nWords[ix]

        startTime = float(lines[4+ix].split(" ")[1])
        endTime = float(lines[4+ix+numWords-1].split(" ")[2])
        sampFreq, audio = wavfile.read(audioFile)
        inputAudio = audio[int(sampFreq*startTime):int(sampFreq*endTime)]
        videoFPS = videoParams["videoFPS"]
        vidInp = np.load(visualFeaturesFile)
        vidInp = vidInp[int(np.floor(videoFPS*startTime)):int(np.ceil(videoFPS*endTime))]


    trgt = [charToIx[char] for char in trgtNWord]
    trgt.append(charToIx["<EOS>"])
    trgt = np.array(trgt)
    trgtLen = len(trgt)


    stftWindow = audioParams["stftWindow"]
    stftWinLen = audioParams["stftWinLen"]
    stftOverlap = audioParams["stftOverlap"]

    if len(inputAudio) < sampFreq*(stftWinLen + 3*(stftWinLen - stftOverlap)):
        padding = int(np.ceil((sampFreq*(stftWinLen + 3*(stftWinLen - stftOverlap)) - len(inputAudio))/2))
        inputAudio = np.pad(inputAudio, padding, "constant")
    inputAudio = inputAudio/np.max(np.abs(inputAudio))

    if noise is not None:
        pos = np.random.randint(0, len(noise)-len(inputAudio)+1)
        noise = noise[pos:pos+len(inputAudio)]
        noise = noise/np.max(np.abs(noise))
        gain = 10**(noiseSNR/10)
        noise = noise*np.sqrt(np.sum(inputAudio**2)/(gain*np.sum(noise**2)))
        inputAudio = inputAudio + noise

    inputAudio = inputAudio/np.sqrt(np.sum(inputAudio**2)/len(inputAudio))

    _, _, stftVals = signal.stft(inputAudio, sampFreq, window=stftWindow, nperseg=sampFreq*stftWinLen, noverlap=sampFreq*stftOverlap,
                                 boundary=None, padded=False)
    audInp = np.abs(stftVals)
    audInp = audInp.T


    if len(audInp)/4 >= len(vidInp):
        inpLen = int(np.ceil(len(audInp)/4))
        leftPadding = int(np.floor((4*inpLen - len(audInp))/2))
        rightPadding = int(np.ceil((4*inpLen - len(audInp))/2))
        audInp = np.pad(audInp, ((leftPadding,rightPadding),(0,0)), "constant")
        leftPadding = int(np.floor((inpLen - len(vidInp))/2))
        rightPadding = int(np.ceil((inpLen - len(vidInp))/2))
        vidInp = np.pad(vidInp, ((leftPadding,rightPadding),(0,0)), "constant")
    else:
        inpLen = len(vidInp)
        leftPadding = int(np.floor((4*inpLen - len(audInp))/2))
        rightPadding = int(np.ceil((4*inpLen - len(audInp))/2))
        audInp = np.pad(audInp, ((leftPadding,rightPadding),(0,0)), "constant")


    reqInpLen = req_input_length(trgt)
    if inpLen < reqInpLen:
        leftPadding = int(np.floor((reqInpLen - inpLen)/2))
        rightPadding = int(np.ceil((reqInpLen - inpLen)/2))
        audInp = np.pad(audInp, ((4*leftPadding,4*rightPadding),(0,0)), "constant")
        vidInp = np.pad(vidInp, ((leftPadding,rightPadding),(0,0)), "constant")

    inpLen = len(vidInp)


    audInp = torch.from_numpy(audInp)
    vidInp = torch.from_numpy(vidInp)
    inp = (audInp,vidInp)
    inpLen = torch.tensor(inpLen)
    trgt = torch.from_numpy(trgt)
    trgtLen = torch.tensor(trgtLen)

    return inp, trgt, inpLen, trgtLen



def collate_fn(dataBatch):
    """
    Collate function definition used in Dataloaders.
    """
    inputBatch = (pad_sequence([data[0][0] for data in dataBatch]),
                  pad_sequence([data[0][1] for data in dataBatch]))
    if not any(data[1] is None for data in dataBatch):
        targetBatch = torch.cat([data[1] for data in dataBatch])
    else:
        targetBatch = None

    inputLenBatch = torch.stack([data[2] for data in dataBatch])
    if not any(data[3] is None for data in dataBatch):
        targetLenBatch = torch.stack([data[3] for data in dataBatch])
    else:
        targetLenBatch = None

    return inputBatch, targetBatch, inputLenBatch, targetLenBatch



def req_input_length(trgt):
    """
    Function to calculate the minimum required input length from the target.
    Req. Input Length = No. of unique chars in target + No. of repeats in repeated chars (excluding the first one)
    """
    reqLen = len(trgt)
    lastChar = trgt[0]
    for i in range(1, len(trgt)):
        if trgt[i] != lastChar:
            lastChar = trgt[i]
        else:
            reqLen = reqLen + 1
    return reqLen

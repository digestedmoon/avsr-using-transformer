import os as _os

args = dict()

# Paths are computed relative to this file so the project works on any machine
_HERE = _os.path.dirname(_os.path.abspath(__file__))

args["CODE_DIRECTORY"]        = _HERE
args["DATA_DIRECTORY"]        = _os.path.join(_HERE, "demo_clips")
args["DEMO_DIRECTORY"]        = _os.path.join(_HERE, "demo_clips")
args["PRETRAINED_MODEL_FILE"] = None
args["TRAINED_MODEL_FILE"]    = "/final/models/audio-visual.pt"   # prepended with CODE_DIRECTORY in app.py
args["TRAINED_LM_FILE"]       = _os.path.join(_HERE, "final", "models", "language_model.pt")
args["TRAINED_FRONTEND_FILE"] = _os.path.join(_HERE, "final", "models", "visual_frontend.pt")


args["PRETRAIN_VAL_SPLIT"] = 0.01
args["NUM_WORKERS"] = 8
args["PRETRAIN_NUM_WORDS"] = 1
args["MAIN_REQ_INPUT_LENGTH"] = 145
args["CHAR_TO_INDEX"] = {" ":1, "'":22, "1":30, "0":29, "3":37, "2":32, "5":34, "4":38, "7":36, "6":35, "9":31, "8":33,
                         "A":5, "C":17, "B":20, "E":2, "D":12, "G":16, "F":19, "I":6, "H":9, "K":24, "J":25, "M":18,
                         "L":11, "O":4, "N":7, "Q":27, "P":21, "S":8, "R":10, "U":13, "T":3, "W":15, "V":23, "Y":14,
                         "X":26, "Z":28, "<EOS>":39}
args["INDEX_TO_CHAR"] = {1:" ", 22:"'", 30:"1", 29:"0", 37:"3", 32:"2", 34:"5", 38:"4", 36:"7", 35:"6", 31:"9", 33:"8",
                         5:"A", 17:"C", 20:"B", 2:"E", 12:"D", 16:"G", 19:"F", 6:"I", 9:"H", 24:"K", 25:"J", 18:"M",
                         11:"L", 4:"O", 7:"N", 27:"Q", 21:"P", 8:"S", 10:"R", 13:"U", 3:"T", 15:"W", 23:"V", 14:"Y",
                         26:"X", 28:"Z", 39:"<EOS>"}


args["NOISE_PROBABILITY"] = 0.25
args["NOISE_SNR_DB"] = 0
args["STFT_WINDOW"] = "hamming"
args["STFT_WIN_LENGTH"] = 0.040
args["STFT_OVERLAP"] = 0.030


args["VIDEO_FPS"] = 25
args["ROI_SIZE"] = 112
args["NORMALIZATION_MEAN"] = 0.4161
args["NORMALIZATION_STD"] = 0.1688


args["SEED"] = 19220297
args["BATCH_SIZE"] = 32
args["STEP_SIZE"] = 16384
args["NUM_STEPS"] = 1000
args["SAVE_FREQUENCY"] = 10


args["INIT_LR"] = 1e-4
args["FINAL_LR"] = 1e-6
args["LR_SCHEDULER_FACTOR"] = 0.5
args["LR_SCHEDULER_WAIT"] = 20
args["LR_SCHEDULER_THRESH"] = 0.001
args["MOMENTUM1"] = 0.9
args["MOMENTUM2"] = 0.999
args["AUDIO_ONLY_PROBABILITY"] = 0.2
args["VIDEO_ONLY_PROBABILITY"] = 0.2


args["AUDIO_FEATURE_SIZE"] = 321
args["NUM_CLASSES"] = 40


args["PE_MAX_LENGTH"] = 2500
args["TX_NUM_FEATURES"] = 512
args["TX_ATTENTION_HEADS"] = 8
args["TX_NUM_LAYERS"] = 6
args["TX_FEEDFORWARD_DIM"] = 2048
args["TX_DROPOUT"] = 0.1


args["BEAM_WIDTH"] = 100
args["LM_WEIGHT_ALPHA"] = 0.5
args["LENGTH_PENALTY_BETA"] = 0.1
args["THRESH_PROBABILITY"] = 0.0001
args["USE_LM"] = True


args["TEST_DEMO_DECODING"] = "greedy"
args["TEST_DEMO_NOISY"] = False
args["TEST_DEMO_MODE"] = "AV"


if __name__ == "__main__":

    for key,value in args.items():
        print(str(key) + " : " + str(value))

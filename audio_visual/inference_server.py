"""
Deep AVSR Inference Server
--------------------------
A FastAPI server that exposes the trained Audio-Visual Speech Recognition model
as a REST API endpoint.

Usage:

Video requirements:
    - Format: .mp4
    - FPS: 25
    - Resolution: 160x160 (face/mouth approximately centered)
    - Audio: Mono, 16000 Hz sample rate
    - Max length: ~6 seconds (transcription must be < 100 characters)
"""

import os
import sys
import uuid
import shutil
import tempfile

import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import args
from models.av_net import AVNet
from models.lrs2_char_lm import LRS2CharLM
from models.visual_frontend import VisualFrontend
from data.utils import prepare_main_input, collate_fn
from utils.preprocessing import preprocess_sample
from utils.decoders import ctc_greedy_decode, ctc_search_decode
from utils.auto_crop import auto_preprocess_video



app = FastAPI(
    title="Deep AVSR Inference API",
    description="Audio-Visual Speech Recognition model — transcribes lip-sync video clips.",
    version="1.0.0",
)

_model = None
_vf = None
_lm = None
_device = None


@app.on_event("startup")
def load_models():
    """Load all model weights once at server startup."""
    global _model, _vf, _lm, _device

    np.random.seed(args["SEED"])
    torch.manual_seed(args["SEED"])

    gpu_available = torch.cuda.is_available()
    _device = torch.device("cuda" if gpu_available else "cpu")
    print(f"\n[AVSR] Using device: {_device}")

    model_path = args["CODE_DIRECTORY"] + args["TRAINED_MODEL_FILE"]
    print(f"[AVSR] Loading AVNet from: {model_path}")
    _model = AVNet(
        args["TX_NUM_FEATURES"],
        args["TX_ATTENTION_HEADS"],
        args["TX_NUM_LAYERS"],
        args["PE_MAX_LENGTH"],
        args["AUDIO_FEATURE_SIZE"],
        args["TX_FEEDFORWARD_DIM"],
        args["TX_DROPOUT"],
        args["NUM_CLASSES"],
    )
    _model.load_state_dict(torch.load(model_path, map_location=_device))
    _model.to(_device)
    _model.eval()
    print("[AVSR] AVNet loaded ✓")

    print(f"[AVSR] Loading VisualFrontend from: {args['TRAINED_FRONTEND_FILE']}")
    _vf = VisualFrontend()
    _vf.load_state_dict(torch.load(args["TRAINED_FRONTEND_FILE"], map_location=_device))
    _vf.to(_device)
    _vf.eval()
    print("[AVSR] VisualFrontend loaded ✓")

    print(f"[AVSR] Loading LM from: {args['TRAINED_LM_FILE']}")
    _lm = LRS2CharLM()
    _lm.load_state_dict(torch.load(args["TRAINED_LM_FILE"], map_location=_device))
    _lm.to(_device)
    _lm.eval()
    if not args["USE_LM"]:
        _lm = None
    print(f"[AVSR] Language Model loaded (USE_LM={args['USE_LM']}) ✓")

    print("\n[AVSR] All models ready. Server is up.\n")



@app.get("/health")
def health_check():
    """Returns server status and whether models are loaded."""
    return {
        "status": "ok",
        "models_loaded": _model is not None,
        "device": str(_device),
        "mode": args["TEST_DEMO_MODE"],
        "decoding": args["TEST_DEMO_DECODING"],
        "use_lm": args["USE_LM"],
    }


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """
    Transcribe speech from an uploaded .mp4 video file.

    Returns:
        JSON with "transcription" key containing the predicted text (uppercase, no punctuation
        except apostrophe).

    Raises:
        400 if the file is not .mp4
        500 on any processing or model error
    """
    if not file.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only .mp4 files are supported.")

    tmp_dir = tempfile.mkdtemp()
    try:
        clip_id = str(uuid.uuid4())
        raw_video_path = os.path.join(tmp_dir, clip_id + "_raw.mp4")
        
        with open(raw_video_path, "wb") as f:
            content = await file.read()
            f.write(content)

        formatted_video_path = os.path.join(tmp_dir, clip_id)
        formatted_video_file = formatted_video_path + ".mp4"
        
        success = auto_preprocess_video(raw_video_path, formatted_video_file)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to format and crop the uploaded video.")

        params = {
            "roiSize": args["ROI_SIZE"],
            "normMean": args["NORMALIZATION_MEAN"],
            "normStd": args["NORMALIZATION_STD"],
            "vf": _vf,
        }
        try:
            preprocess_sample(formatted_video_path, params)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Base AVSR preprocessing failed: {str(e)}")

        audio_file = formatted_video_path + ".wav"
        visual_features_file = formatted_video_path + ".npy"

        if not os.path.exists(audio_file):
            raise HTTPException(status_code=500, detail="Audio extraction failed. Make sure ffmpeg is installed.")
        if not os.path.exists(visual_features_file):
            raise HTTPException(status_code=500, detail="Visual feature extraction failed.")

        audio_params = {
            "stftWindow": args["STFT_WINDOW"],
            "stftWinLen": args["STFT_WIN_LENGTH"],
            "stftOverlap": args["STFT_OVERLAP"],
        }
        video_params = {"videoFPS": args["VIDEO_FPS"]}

        inp, _, inp_len, _ = prepare_main_input(
            audio_file,
            visual_features_file,
            None,
            None,
            args["MAIN_REQ_INPUT_LENGTH"],
            args["CHAR_TO_INDEX"],
            args["NOISE_SNR_DB"],
            audio_params,
            video_params,
        )
        input_batch, _, input_len_batch, _ = collate_fn([(inp, None, inp_len, None)])

        input_batch = (
            (input_batch[0].float()).to(_device),
            (input_batch[1].float()).to(_device),
        )
        input_len_batch = (input_len_batch.int()).to(_device)

        mode = args["TEST_DEMO_MODE"]
        if mode == "AO":
            input_batch = (input_batch[0], None)
        elif mode == "VO":
            input_batch = (None, input_batch[1])
        elif mode != "AV":
            raise HTTPException(status_code=500, detail=f"Invalid TEST_DEMO_MODE: {mode}")

        with torch.no_grad():
            output_batch = _model(input_batch)

        if args["TEST_DEMO_DECODING"] == "greedy":
            pred_batch, _ = ctc_greedy_decode(output_batch, input_len_batch, args["CHAR_TO_INDEX"]["<EOS>"])

        elif args["TEST_DEMO_DECODING"] == "search":
            beam_params = {
                "beamWidth": args["BEAM_WIDTH"],
                "alpha": args["LM_WEIGHT_ALPHA"],
                "beta": args["LENGTH_PENALTY_BETA"],
                "threshProb": args["THRESH_PROBABILITY"],
            }
            pred_batch, _ = ctc_search_decode(
                output_batch,
                input_len_batch,
                beam_params,
                args["CHAR_TO_INDEX"][" "],
                args["CHAR_TO_INDEX"]["<EOS>"],
                _lm,
            )
        else:
            raise HTTPException(status_code=500, detail=f"Invalid TEST_DEMO_DECODING: {args['TEST_DEMO_DECODING']}")

        pred = pred_batch[:][:-1]
        transcription = "".join([args["INDEX_TO_CHAR"][ix] for ix in pred.tolist()])

        return JSONResponse(content={"transcription": transcription, "mode": mode})

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("inference_server:app", host="0.0.0.0", port=8000, reload=False)

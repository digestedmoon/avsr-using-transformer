# AVSR using Transformer

A implementation of the **Audio-Visual Speech Recognition** (AVSR) model based on the [Deep Audio-Visual Speech Recognition](https://arxiv.org/abs/1809.02108) paper. Uses a Transformer (TM-CTC) architecture trained on the LRS2 dataset.

## Project Structure

```
audio_visual/
├── app.py                  # Flask web server (main entry point)
├── config.py               # Hyperparameters & paths (all paths are relative — works on any machine)
├── demo.py                 # CLI demo script
├── models/
│   ├── av_net.py           # TM-CTC Audio-Visual network
│   ├── visual_frontend.py  # CNN-based visual frontend
│   └── lrs2_char_lm.py     # Character-level language model
├── data/
│   ├── utils.py            # Dataset preparation helpers
│   └── lrs2_dataset.py     # LRS2 dataset class
├── utils/
│   ├── auto_crop.py        # Automatic face crop & video formatting
│   ├── preprocessing.py    # Extract audio & visual features from video
│   ├── decoders.py         # Greedy & beam-search CTC decoders
│   ├── general.py          # Training helpers
│   └── metrics.py          # CER / WER computation
├── static/                 # CSS & JS for the web UI
├── templates/              # Jinja2 HTML templates
└── final/
    └── models/             # ← Place downloaded weights here (see below)
```

## Model Weights

> Email me

After downloading, place the three `.pt` files in `audio_visual/final/models/`:

```
audio_visual/final/models/
├── audio-visual.pt    
├── language_model.pt 
└── visual_frontend.pt   
```

## Requirements

**System:**
```bash
# ffmpeg is required for audio/video processing
sudo apt install ffmpeg     # Ubuntu/Debian
sudo dnf install ffmpeg     # Fedora
```

**Python packages:**
```bash
pip install -r requirements.txt
```

## Running the Flask Server

```bash
cd audio_visual
python app.py
```

Then open **http://localhost:5000** in your browser. Upload an `.mp4` video file and click **Transcribe**.

## CLI Demo

Place `.mp4` clips in `audio_visual/demo_clips/`, then:

```bash
cd audio_visual
python demo.py
```

## Configuration

All settings are in `audio_visual/config.py`. Key options for inference:

| Key | Default | Description |
|-----|---------|-------------|
| `TEST_DEMO_MODE` | `"AV"` | `"AV"` (audio+visual), `"AO"` (audio-only), `"VO"` (video-only) |
| `TEST_DEMO_DECODING` | `"greedy"` | `"greedy"` or `"search"` (beam search with LM) |
| `USE_LM` | `True` | Use language model during beam search |

## Results (LRS2 Test Set)

| Mode | Greedy WER | Beam+LM WER |
|------|-----------|-------------|
| AV (clean) | 10.3% | **6.8%** |
| AV (noisy 0dB) | 29.1% | 22.1% |

## References

- [Deep Audio-Visual Speech Recognition, Afouras et al. 2018](https://arxiv.org/abs/1809.02108)
- CTC beam search adapted from [Harald Scheidl, CTCDecoder](https://github.com/githubharald/CTCDecoder)

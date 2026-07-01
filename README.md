# Generative AI Hub

A small collection of concise, educational implementations and example projects for generative models across audio, image, text→image, text→video, and video summarization — implemented in Python with PyTorch. Each top-level folder contains a minimal pipeline: dataset, model definitions, training loop, and a runnable entrypoint.

## Quick Summary

- Modalities: audio (TTS / vocoder), image (GAN), text→image, text→video experiments, and video summarization.
- Purpose: learning reference implementations and reproducible experiments, not production-ready systems.
- Who it's for: researchers, students, and engineers who want compact example code to experiment with generative models.

### Stack
- **Language(s):** Python 3.8+
- **Framework / runtime:** PyTorch (torch, torchvision)
- **Notable libraries:** torch, torchvision, matplotlib, soundfile

## Repository Layout (Top-level)

```
audio-generation/            # TTS & vocoder examples (FastSpeech2 + HiFiGAN)
  dataset.py                 # dataset builders, vocabulary, constants
  acoustic_model.py          # FastSpeech2-style acoustic model
  vocoder_model.py           # HiFiGAN-style vocoder
  train.py                   # training loops for acoustic + vocoder
  inference.py               # end-to-end synthesis helpers
  main.py                    # example orchestration (train + inference)
  utils.py                   # helpers, collate functions

image_generation/            # Simple GAN example
  dataset.py                 # dataset builder for images
  generator.py               # Generator network
  discriminator.py           # Discriminator network
  train.py                   # GAN training loop
  main.py                    # run training / visualize outputs
  utils.py                   # weight init, helpers

text-to-imagegeneration/     # Text-to-image experiment (GAN-style / conditional)
  dataset.py
  generator.py
  discriminator.py
  train.py
  main.py
  transform.py               # text/image transforms
  utils.py

text-to-video-generation/    # Experimental text→video pipeline
  dataset.py
  model.py                   # model definitions and helper layers
  layers.py
  train.py
  main.py
  utils.py
  vocabulary.py              # token/vocabulary utilities

video-summarization/         # Video summarization and multimodal utilities
  dataset.py
  model.py
  layers.py
  train.py
  main.py
  inference.py               # example inference / summarization pipeline
  utils.py
README.md
```

## How It Fits Together

Each subproject follows the same pattern: `dataset.py` builds datasets and DataLoaders, model files (e.g., `generator.py`, `acoustic_model.py`) define the networks, `train.py` implements training/validation loops, and `main.py` ties them together and saves outputs (generated_images/ or generated_audio/). The audio pipeline contains both an acoustic model (mel spectrogram predictor) and a vocoder for waveform synthesis; `inference.py` connects them for end-to-end synthesis.

## How To Get Started (Short Path)

1. Clone:
   ```bash
   git clone https://github.com/krishnashashanth-sks/generative-ai-hub.git
   cd generative-ai-hub
   ```

2. Create a virtual environment and install core deps (example):
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   pip install --upgrade pip
   pip install torch torchvision matplotlib soundfile
   ```
   Note: For GPU support install the appropriate CUDA build of PyTorch per https://pytorch.org.

3. Run an example (each subfolder has a runnable `main.py`):
   - Audio example (train acoustic + vocoder, then run inference):
     ```bash
     cd audio-generation
     python main.py
     # outputs saved to generated_audio/ by default
     ```
   - Image GAN example:
     ```bash
     cd ../image_generation
     python main.py
     # outputs saved to generated_images/
     ```
   - Text→image / Text→video / Video summarization:
     ```bash
     cd ../text-to-imagegeneration
     python main.py

     cd ../text-to-video-generation
     python main.py

     cd ../video-summarization
     python main.py
     ```

4. If a subproject includes a `train.py`, you can call the training functions directly or inspect `main.py` for example training hyperparameters and data paths.

## Expected Outputs
- audio-generation: a directory `generated_audio/` with synthesized .wav files (inference example in `main.py`).
- image_generation: a directory `generated_images/` with example generated images and loss plots.
- text->* projects: model checkpoints, intermediate visualizations, and sample outputs depending on the subproject.

## Recommendations (Improvements You May Want)
- Add per-subproject `requirements.txt` files and a top-level `environment.yml` or `requirements.txt`.
- Add short README files inside each top-level folder describing dataset source, expected data layout, and minimal commands to run training/inference.
- Add example (or scripts) to download or point to pretrained checkpoints for faster demos.
- Add unit tests or smoke tests that run a single training iteration on dummy data to validate environments.
- Add a LICENSE file (choose an OSI-approved license if you want others to reuse).

## Contributing

Contributions welcome. Please:
1. Open an issue to discuss large changes.
2. Add a short README for any new subproject you add with instructions and dependencies.
3. Keep examples small and focused — each subproject should be runnable on a single-GPU or CPU for quick testing.

## Notes & Caveats
- These are educational/simple reference implementations. They prioritize clarity and compactness over production performance, distributed training, or advanced checkpointing.
- Some datasets and long-running experiments may not be included; adjust `dataset.py` or point to local data as needed.

## Contact

Created by krishnashashanth-sks. Open an issue or PR if you want help converting a subproject into a tested example or adding a requirements file, demo notebooks, or downloadable checkpoints.

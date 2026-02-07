# Clip Analyzer - Backend

This service is responsible for exposing a REST API as the main entry point for
the video clip processing.

## Tech Stack

This project uses:

### Core Framework

- [Python](https://www.python.org/) - core programming language
- [FastAPI](https://fastapi.tiangolo.com/) - web framework for building the APIs
- [Pydantic](https://docs.pydantic.dev/latest/) - type-safety and data
  validation

### AI & Machine Learning

- [PyTorch](https://pytorch.org/) - deep-learning framework
- [HuggingFace](https://huggingface.co/) - foundational ML tooling

## Models

- [Moondream2](https://huggingface.co/vikhyatk/moondream2) - small vision
  language model (VLM)
- [Microsoft Phi-3 Mini](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) -
  small language model (SLM)

### Video Processing

- [OpenCV](https://opencv.org/) - computer vision library
- [yt-dlp](https://github.com/yt-dlp/yt-dlp): feature-rich video downloader

## Local Environment

To run this application in your machine, follow the instructions below:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
python -m app.main
```

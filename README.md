# Twitch Clip AI Analyzer

This project is a small-scoped application to analyze the visual contents of
short videos (in this case, clips from the Twitch Platform).

I developed this application as a learning project to learn more about applying
modern AI Engineering techniques.

## Motivation

The main goal of this project was to deepen my understanding of AI engineering
while applying modern Python tooling and architectural techniques to machine
learning and artificial intelligence.

For the development of this project, I decided to specifically focus on
practicing the following:

- **FastAPI:** Building robust, type-safe APIs.
- **Local AI Inference:** Running Large Language Models (LLMs) and
  Vision-Language Models (VLMs) locally on consumer hardware.
- **AI Pipelines:** Orchestrating multi-step AI tasks (Video -> Frames -> Vision
  Analysis -> Text Summarization).

## What It Does

This application takes an URL from a Twitch Clip and performs a complete
automated analysis:

1. **Downloads** the clip and processes the video stream.
2. **Extracts** key frames at specific intervals.
3. **Analyzes** each frame using **Moondream2** (VLM) to understand the visual
   context.
4. **Summarizes** a cohesive description using **Microsoft Phi-3** (SLM) to
   describe "what happened" in the clip based on the visual data.

## The AI Pipeline

The core of the application relies on a two-step inference process:

- **Moondream2:** A small, efficient computer vision model that "looks" at
  individual frames and describes them in natural language.
- **Phi-3 Mini:** A reasoning model that takes each individual description from
  the extracted frames and compile them into an overview summary.

## Project Structure

> [!WARNING]
> The application UI is not available yet.

- **`/backend`**: The core logic, API, and AI services.
- **`/frontend`**: An interative UI for users to access it with ease in the
  local environment.

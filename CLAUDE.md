# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development Setup
- `/install`: Create virtual environment and install dependencies
  ```bash
  python -m venv venv && source venv/bin/activate && pip install -e .
  ```
- `/build`: Install with CUDA support for GPU acceleration
  ```bash
  pip install torch==2.3.0+cu118 torchaudio==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
  ```

### Application Entry Points
- `/webui`: Launch main Gradio WebUI interface
  ```bash
  f5-tts_webui
  # or python src/f5_tts/f5_tts_webui.py
  ```
- `/api`: Start REST API server
  ```bash
  python src/f5_tts/api_new.py --server
  ```
- `/train`: Launch finetune training interface
  ```bash
  f5-tts_finetune-gradio
  ```
- `/infer`: Run CLI inference
  ```bash
  f5-tts_infer-cli --help
  ```

### Code Quality
- `/lint`: Check and fix code style issues
  ```bash
  ruff check . --fix && ruff format .
  ```
- `/test`: Manual testing via WebUI or API examples
  ```bash
  python examples/multiline_text_example.py
  python examples/multistyle_modes_example.py
  ```

## Architecture

### High-Level Structure
F5-TTS Thai is a Thai language Text-to-Speech system built on Flow Matching technology. The codebase is organized into several key modules:

- **Core Models** (`src/f5_tts/model/`): DiT, UNetT, MMDiT architectures with CFM (Conditional Flow Matching)
- **Inference** (`src/f5_tts/infer/`): CLI, Gradio WebUI, and utility functions for TTS generation
- **Training** (`src/f5_tts/train/`): Finetuning capabilities with both CLI and Gradio interfaces
- **Thai Language Support** (`src/f5_tts/cleantext/`): Thai-specific text preprocessing and tokenization
- **APIs** (`src/f5_tts/api*.py`): REST API server with multi-style speech generation

### Model Flow
1. Text preprocessing (Thai number conversion, repetition handling)
2. Tokenization via pythainlp for Thai language
3. Flow matching generation through DiT/UNetT/MMDiT backbones
4. Vocoder-based audio synthesis
5. Post-processing (silence removal, audio normalization)

### Thai Language Components
- Custom Thai number-to-text conversion (`cleantext/number_tha.py`)
- Thai text repetition processing (`cleantext/th_repeat.py`) 
- Integration with pythainlp for proper Thai tokenization
- Whisper API integration for Thai ASR (`utils/whisper_api.py`)

## Development Notes

### Code Style
- Uses Ruff for linting and formatting (120 char line length, Python 3.10+ target)
- Single-line imports enforced with 2 blank lines after imports
- Entry points defined in `pyproject.toml` for CLI tools

### Key Dependencies
- PyTorch 2.0+ with CUDA 11.8 support for GPU acceleration
- Gradio for WebUI interfaces
- pythainlp for Thai language processing
- Accelerate for distributed training

### Windows Support  
- Batch scripts provided: `install.bat`, `app-webui.bat`, `train.bat`
- Virtual environment activation handled automatically in batch files

### Testing Strategy
- No formal test suite - relies on manual testing via WebUI and API examples
- Use `/examples` directory for API testing scenarios
- GPU/CPU compatibility testing via different model configurations
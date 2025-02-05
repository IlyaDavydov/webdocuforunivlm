# Requirements & Installation

Before getting started with our pipeline, make sure you have the necessary dependencies installed in your virtual environment.

## Installation

Just use this command to install our pipeline: pip install univlm

You can install all required dependencies at once using:

```bash
pip install -r requirements.txt

Alternatively, install them manually:

### Core Dependencies
The following Python libraries are essential for running the pipeline:

- `transformers` – for working with Hugging Face models  
- `torch` – PyTorch, used for deep learning computations  
- `vllm` – efficient model inference with VLLM  
- `concurrent.futures` – for parallel processing  
- `fuzzywuzzy` – for fuzzy string matching  
- `subprocess` – for executing system commands  
- `json` – handling JSON data  
- `pathlib` – working with filesystem paths  
- `diffusers` – for diffusion-based models  

### Notes
- Requires an internet connection for model downloads.
- Conda and Git must be pre-installed.
- Some methods use parallel processing for improved performance.
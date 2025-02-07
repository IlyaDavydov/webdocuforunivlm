# Exclusive Models Documentation

## Table of Contents

1. [Apple Depth Pro](#apple-depth-pro)

---

## Apple Depth Pro

### About
Apple Depth Pro is a cutting-edge depth estimation model designed to leverage Apple’s hardware and software capabilities for high-precision depth mapping. It integrates seamlessly with Apple’s vision framework to provide real-time depth perception for various applications.

### Requirements
- macOS 12.0 or later
- Xcode 13 or later
- Python 3.8+
- PyTorch 1.10+
- NumPy
- OpenCV
- Apple Vision Framework
- certifi==2024.12.14
- charset-normalizer==3.4.1
- contourpy==1.3.0
- cycler==0.12.1
- -e git+https://github.com/apple/ml-depth-pro.git@b2cd0d51daa95e49277a9f642f7fd736b7f9e91d#egg=depth_pro
- filelock==3.17.0
- fonttools==4.55.7
- fsspec==2024.12.0
- huggingface-hub==0.28.0
- idna==3.10
- importlib_resources==6.5.2
- Jinja2==3.1.5
- kiwisolver==1.4.7
- MarkupSafe==3.0.2
- matplotlib==3.9.4
- mpmath==1.3.0
- networkx==3.2.1
- numpy==1.26.4
- nvidia-cublas-cu12==12.4.5.8
- nvidia-cuda-cupti-cu12==12.4.127
- nvidia-cuda-nvrtc-cu12==12.4.127
- nvidia-cuda-runtime-cu12==12.4.127
- nvidia-cudnn-cu12==9.1.0.70
- nvidia-cufft-cu12==11.2.1.3
- nvidia-curand-cu12==10.3.5.147
- nvidia-cusolver-cu12==11.6.1.9
- nvidia-cusparse-cu12==12.3.1.170
- nvidia-nccl-cu12==2.21.5
- nvidia-nvjitlink-cu12==12.4.127
- nvidia-nvtx-cu12==12.4.127
- packaging==24.2
- pillow==11.1.0
- pillow_heif==0.21.0
- pyparsing==3.2.1
- python-dateutil==2.9.0.post0
- PyYAML==6.0.2
- regex==2024.11.6
- requests==2.32.3
- safetensors==0.5.2
- six==1.17.0
- sympy==1.13.1
- timm==1.0.14
- tokenizers==0.21.0
- torch==2.5.1
- torchvision==0.20.1
- tqdm==4.67.1
- transformers==4.48.1
- triton==3.1.0
- typing_extensions==4.12.2
- urllib3==2.3.0
- zipp==3.21.0

### Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/example/apple-depth-pro.git
   cd apple-depth-pro
   ```
2. Create a virtual environment and install dependencies:
   ```sh
   python3 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Build and run the project:
   ```sh
   python main.py
   ```

### Example Code
```python
import cv2
import numpy as np
from depth_model import AppleDepthPro

# Load the model
depth_model = AppleDepthPro()

# Read an input image
image = cv2.imread("input.jpg")

# Estimate depth
depth_map = depth_model.estimate_depth(image)

# Display depth map
cv2.imshow("Depth Map", depth_map)
cv2.waitKey(0)
cv2.destroyAllWindows()
```


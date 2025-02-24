# Univlm API Reference

## Yggdrasil Class

Core class for unified vision-language model management. Supports multiple model types (VLLM, HuggingFace, Exclusive).

### `__init__(model_name, Feature_extractor, Image_processor, Config_Name=None)`

Initialize Yggdrasil pipeline instance.

**Parameters**:
- `model_name` (str): Identifier for model to load
- `Feature_extractor` (bool): Whether to use feature extraction
- `Image_processor` (bool): Whether to process images
- `Config_Name` (str, optional): Specific configuration name for HuggingFace models

### Methods

#### `load() -> str`
Attempt to load model using supported backends (VLLM → HuggingFace → Exclusive).

**Returns**:
- Loading status string: "Loaded" or "Failed to Load"

**Behavior**:
1. Tries VLLM backend first
2. Falls back to HuggingFace models
3. Attempts exclusive model loading as last resort
4. Prints loading diagnostics

#### `Proccessor() -> str`
Load appropriate data processor for loaded model.

**Returns**:
- "Processor Loaded" on success

**Behavior**:
- Auto-detects processor type based on model backend
- Handles HuggingFace processors (tokenizers/image processors)
- Skips processing for VLLM/Exclusive models

#### `inference(payload: dict) -> Union[str, list]`
Execute inference on input payload.

**Parameters**:
- `payload` (dict): Input data with flexible keys:
  - Text keys: ["prompt", "text", "input_text", "inputs"]
  - Image keys: ["images", "image", "pixel_values", "visual_input"]

**Returns**:
- Single result (str) for single input
- List of results for batch inputs

**Payload Examples**:
```python
# Single text input
{"prompt": "Describe this image"}

# Batch image processing 
{"images": [img1, img2], "text": ["Caption A", "Caption B"]}

# Univlm API Reference

## Yggdrasil Class

Core class for unified vision-language model management. Supports multiple model types (VLLM, HuggingFace, Exclusive).

### `__init__(model_name, Feature_extractor, Image_processor, Config_Name=None)`

Initialize Yggdrasil pipeline instance.

| Parameter           | Type    | Description                              |
|---------------------|---------|------------------------------------------|
| `model_name`        | str     | Model identifier                         |
| `Feature_extractor` | bool    | Enable feature extraction                |
| `Image_processor`   | bool    | Enable image processing                  |
| `Config_Name`       | str     | HF config name (optional)                | 

### Methods

#### `load() -> str`
Attempt to load model using supported backends (VLLM → HuggingFace → Exclusive).

## Returns:  
- `"Loaded"`
- `"Failed to Load"`

## Behavior:
- **Tries VLLM backend first**
- **Falls back to HuggingFace models**
- **Attempts exclusive model loading as last resort**
- **Prints loading diagnostics**

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

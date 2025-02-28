# Univlm API Reference

## Unify Class

Core class for unified vision-language model management. Supports multiple model types (VLLM, HuggingFace, Exclusive).

### `__init__(model_name, Feature_extractor, Image_processor, Config_Name=None)`

Initialize unify pipeline instance.

| Parameter           | Type    | Description                              |
|---------------------|---------|------------------------------------------|
| `model_name`        | str     | Model identifier                         |
| `Feature_extractor` | bool    | Enable feature extraction                |
| `Image_processor`   | bool    | Enable image processing                  |
| `Config_Name`       | str     | HF config name (optional)                | 

### Methods

#### `load() -> str`
**Description:** Loads model using priority: VLLM → HF → Exclusive.

**Returns:**

- `str`: Loading status ("Loaded" or "Failed to Load")

**Behavior:**

- Attempts VLLM loading with GPU memory utilization: 90%, Max sequence length: 2048
- Falls back to HF via `HFModelSearcher`: Handles config selection via CLI when ambiguous, uses `reference_table` for model class mapping
- For Exclusive models: Calls `env_setup()` and `load_model()`
  
**Example:**

```python
model = unify("gpt2", feature_extractor, image_processor)
status = model.load()
```

#### `Proccessor() -> str`
**Description:** Determines the appropriate processor (Tokenizer or Processor) for the model

| Parameter           | Type    | Description                              |
|---------------------|---------|------------------------------------------|
| `model_name`        | str     | Name of the model to process             |

**Returns**:

- Type of processor selected ('Processor' or 'Tokenizer')

**Raises**:

- ValueError: If model not loaded

**Behavior**:

- For HF: Uses HFProcessorSearcher
- Skips for VLLM/Exclusive models
- Requires prior model loading

#### `_standardize_payload(self, payload) -> tuple[dict, bool]`

| Parameter           | Type    | Description                              |
|---------------------|---------|------------------------------------------|
| `payload`           | dict    | Raw data with potential aliases for keys |

**Description:** Standardizes input payload keys for compatibility with both VLLM and HF backends. Handles both single inputs and batches.

**Parameters:**

- `payload` (dict): Raw input data with potential aliases for keys.

**Returns:**

- `tuple[dict, bool]`: A tuple containing:
  - `standardized` (dict): Normalized payload with keys `"text"` and optionally `"pixel_values"`. Values are always lists.
  - `is_batch` (bool): `True` if the input is a batch (multiple items), `False` for single inputs.

**Behavior:**

- Normalizes key aliases for text and image inputs.
- Converts single inputs to lists for consistency.
- Automatically detects batch inputs.

**Example:**
```python
# Single text input
payload = {"prompt": "Hello"}
standardized, is_batch = model._standardize_payload(payload)
# Returns: ({"text": ["Hello"]}, False)

# Batch of images
payload = {"images": [img1, img2]}
standardized, is_batch = model._standardize_payload(payload)
# Returns: ({"pixel_values": [img1, img2]}, True)
```

#### `_get_processor_input_names(processor)`

| Parameter           | Type    | Description                              |
|---------------------|---------|------------------------------------------|
| `processor`         | dict    | The processor object (e.g., tokenizer)   |

**Description:** Determines the correct input parameter names for different processor types.

**Parameters:**

- `processor` (Any): The processor object (e.g., tokenizer, image processor).

**Returns:**

- `dict[str, Optional[str]]`: A dictionary mapping input types (`"text"` and `"image"`) to their corresponding parameter names. If a processor does not support a specific input type, the value will be `None`.

**Behavior:**

- Inspects the processor's class name to determine its type.
- Returns appropriate parameter names based on the processor's capabilities:
  - Multi-modal processors: Use `"text"` for text and `"images"` for images.
  - Tokenizers: Only handle text (`"text"`), with `"image"` set to `None`.
  - Image processors/feature extractors**: Only handle images (`"pixel_values"`), with `"text"` set to `None`.
  - Default fallback: Uses `"text"` and `"pixel_values"` for unknown processor types.

**Example:**
```python
# Tokenizer
tokenizer = Tokenizer()
input_names = model._get_processor_input_names(tokenizer)
# Returns: {"text": "text", "image": None}

# Image processor
image_processor = ImageProcessor()
input_names = model._get_processor_input_names(image_processor)
# Returns: {"text": None, "image": "pixel_values"}
```

#### `inference(payload)`

| Parameter           | Type    | Description                              |
|---------------------|---------|------------------------------------------|
| `payload`           | dict    | Input data containing text, images       |

**Description:** Performs inference on single or batch inputs using the loaded model.

**Parameters:**

- `payload` (dict): Input data containing text, images, or both. Supports batch inputs.

**Returns:**

- `Union[list, Any]`: Inference results. Returns a list for batch inputs or a single result for non-batch inputs.

**Behavior:**

**Input Standardization:**

   - Uses `_standardize_payload` to normalize input keys and detect batch mode.
   - Raises `ValueError` if no valid input keys are found.

**Backend-Specific Handling:**

   - **VLLM:**
     - Requires text input (`"text"`).
     - Uses `SamplingParams` with:
       - Temperature: 0.8
       - Max tokens: 128
       - Stop sequences: `["</s>", "[/INST]", "Assistant:", "Human:"]`
     - Automatically handles batch generation.
   - **HF (Hugging Face):**
     - Ensures the processor is loaded.
     - Processes inputs with dynamic padding.
     - Supports:
       - `AutoModelForCausalLM`
       - `AutoModelForSeq2SeqLM`
       - `AutoModelForVision2Seq`
       - `AutoModelForMaskedLM`
     - Falls back to `model.generate` if direct inference fails.
   - **Exclusive Models:**
     - Processes inputs sequentially.
     - Uses the model's custom `processor` and `infer` methods.

**Output Formatting:**
   - Returns a list for batch inputs.
   - Returns a single result for non-batch inputs.

**Raises:**

- `ValueError`: If no valid input keys are found or if the processor is not loaded (for HF).
- `Exception`: Propagates backend-specific errors during inference.

**Examples:**

**1. VLLM Backend:**
```python
payload = {"prompt": "Explain quantum physics"}
result = model.inference(payload)
# Returns: "Quantum physics is the study of..."
```

**2. HF Backend (Batch Input):**
```python
payload = {
    "text": ["What is AI?", "What is ML?"],
    "images": [img1, img2]
}
results = model.inference(payload)
# Returns: ["AI is...", "ML is..."]
```

**3. Exclusive Backend:**
```python
payload = {"input_text": "Estimate the depth of image", "image": "example.jpg"}
result = model.inference(payload)
# Returns: Estimated depth
```

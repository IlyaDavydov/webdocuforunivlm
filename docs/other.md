# AI Model Library Technical Documentation

## Overview
A comprehensive library providing unified interfaces for AI model operations, including model loading, processing, and inference across multiple frameworks. The library supports Hugging Face models, VLLM-based models, and exclusive models like AppleDepth, with standardized input/output handling and extensive error management.

## Classes

### `unify`
Universal model loader and inference handler supporting multiple model architectures and frameworks.

#### Properties
- `model`: Loaded model instance
- `model_type`: Type identifier ("VLLM", "HF", or "Exclusive")
- `Processor`: Associated processor instance
- `model_name`: Model identifier/path
- `Feature_extractor`: Boolean for feature extractor usage
- `Image_processor`: Boolean for image processor usage
- `Config_Name`: Model configuration name
- `map`: Reference to model class

#### Methods
##### `__init__(model_name, Feature_extractor=False, Image_processor=False, Config_Name=None)`
Initializes model handler with specified parameters.

##### `load()`
Attempts model loading in priority order:
1. VLLM loading with GPU memory optimization
2. Hugging Face model loading with config matching
3. Exclusive model loading with environment setup
Returns "Loaded" on success or "Failed to Load" on failure.

##### `Proccessor()`
Initializes appropriate processor based on model type:
- VLLM: No processor required
- HF: Automatic processor selection and initialization
- Exclusive: Model-specific processor initialization
Returns "Processor Loaded" on success.

##### `_standardize_payload(payload)`
Internal method for input standardization.
- Handles text and image inputs
- Supports single inputs and batches
- Standardizes key names across frameworks

##### `_get_processor_input_names(processor)`
Internal method determining processor parameter names based on type.

##### `inference(payload)`
Performs model inference with extensive input handling:
- Input validation and standardization
- Batch processing support
- Automatic padding and tensor manipulation
- Framework-specific output processing

### `HFModelSearcher`
Utility class for Hugging Face model identification and searching.

#### Properties
- `ordered_dicts_mapping`: Model type mappings
- `model_classes_mapping`: Model class references

#### Methods
##### `extract_model_family(hf_path)`
Extracts core model family name using regex patterns:
- Removes size/version suffixes
- Handles special model families
- Returns normalized model family name

##### `search_in_ordered_dict(mapping_name, ordered_dict, config_name)`
Internal search method for exact config matching.

##### `search(query=None, config=None)`
Advanced model search with:
- Parallel exact matching
- Fuzzy matching fallback
- Configurable search paths
Returns matching model information or None.

### `HFProcessorSearcher`
Advanced processor matching and selection utility.

#### Properties
- `ordered_dicts_mapping`: Processor mappings
- `model_classes_mapping`: Processor class references

#### Methods
##### `extract_model_family(hf_path)`
Specialized processor family name extraction.

##### `search(query, feature_extractor=False, image_processor=False, tokenizer=False)`
Two-phase processor search:
1. Targeted search based on flags
2. Fallback search across all processor types

##### `_get_priority_order(fe, ip, tok)`
Internal method for determining processor search priority.

##### `_search_mappings(mappings, query)`
Internal method for processor mapping search with fuzzy matching.

##### `_select_best_match(matches, priority_order)`
Internal method for optimal processor selection.

### `appledepth`
Implementation for Apple's DepthPro depth estimation model.

#### Properties
- `model`: DepthPro model instance
- `transform`: Image transformation pipeline
- `image`: Processed input image
- `f_px`: Focal length in pixels

#### Methods
##### `download_checkpoints()`
Manages model checkpoint acquisition:
- Checks existing checkpoints
- Creates directories as needed
- Handles huggingface-hub installation
Returns status code (0 for success, 1 for failure).

##### `env_setup()`
Environment initialization:
- Script execution management
- Directory structure setup
- Permission handling
Returns status code.

##### `load_model()`
Model initialization:
- Imports depth_pro module
- Creates model and transforms
- Sets model to evaluation mode

##### `processor(image_path, text=None)`
Image processing pipeline:
- Loads RGB image
- Applies transformations
- Calculates focal length

##### `infer()`
Depth estimation inference:
- Processes transformed image
- Returns predicted depth map

## Dependencies
### Core Libraries
- transformers
- vllm
- torch
- huggingface-hub
- fuzzywuzzy

### Additional Requirements
- subprocess
- concurrent.futures
- re
- os

## Technical Notes
- Model loading prioritizes VLLM for supported models
- Processor selection uses fuzzy matching with 60% threshold
- Batch processing automatically handles variable sequence lengths
- Error handling includes graceful fallbacks and detailed error messages
- Tensor operations are optimized for batch processing
- Model checkpoints are cached to prevent redundant downloads
- Multi-threaded operations for search and matching functions

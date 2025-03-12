# Usage Examples

### **Example: Hugging Face Model (Not VLLM)**
- This example demonstrates how to load a Hugging Face model using `univlm` and perform inference.
```python
from univlm.Model import unify

y = unify("nlptown/bert-base-multilingual-uncased-sentiment", Config_Name="BertForNextSentencePrediction")

y.load()
payload = {"text": "Hello, how are you?", "pixel_values": None}
y.Proccessor()
output = y.inference(payload)
print(output)
```
### **Example of VLM**
- This is an example of model supported on vLLM task with the use of "*Salesforce/blip-vqa-base*"
```python
from univlm.Model import unify  
from PIL import Image
import requests

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
listy = [raw_image, raw_image]
payload = {"pixel_values": listy, "text": ["how many dogs?", "color of dog"]}
y = unify("Salesforce/blip-vqa-base", Config_Name='BlipForQuestionAnswering')
y.load()
y.Proccessor()
output = y.inference(payload)
print(output)
```
### **Example of Image Only task**
- This is an example of image only task with the use of "*facebook/sam-vit-base*"
```python
from univlm.Model import unify

# Image Segmentation with SAM (Vision Model)
img_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
payload = {"pixel_values": image, "text": None}
y = unify("facebook/sam-vit-base", Image_processor=True, Config_Name= 'SamModel')
y.load()
y.Proccessor()
output = y.inference(payload)
print(output)
```
### **VLLM example**
- This is an example of the use of "*facebook/opt-125m*"
```python
from univlm.Model import unify
prompts = ["Hello, my name is", "what is the capital of United States"]
y = unify("facebook/opt-125m")
y.load()
payload = {"text": prompts, "pixel_values": None}
output = y.inference(payload)
print(output)
```
### **Depth Estimation**
- This is an example of Depth Estimation with the use of "*depth-anything-large*"
```python
from univlm.Model import unify  
from PIL import Image
import requests

y = unify("AppledepthPro")
y.load()
y.Proccessor()
image_path = "input.jpg"
output = y.inference(image_path)
print("Depth map generated:", output)

```
### **Object detection example**
```python
from univlm.Model import unify
from PIL import Image
import requests

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

listy = [raw_image, raw_image]
payload = {"pixel_values": listy, "text": ["how many dogs?", "color of dog"]}

y = unify("Salesforce/blip-vqa-base")
y.load()
y.Proccessor()
output = y.inference(payload)
print(output)
```


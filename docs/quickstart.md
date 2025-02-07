# Usage Examples

### **Example: Hugging Face Model (Not VLLM)**
- This example demonstrates how to load a Hugging Face model using `Yggdrasil` and perform inference.
```python
from univlm.Model import unify  

y = unify("nlptown/bert-base-multilingual-uncased-sentiment",Config_Name = 'BertForNextSentencePrediction') # also try not providing Config_Name in cli
y.load()
payload = { "text": "Hello, how are you?", "pixel_values": None }
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
listy = [raw_image,raw_image]
payload = {"pixel_values": listy, "text": ["how many dogs?","color of dog"]}

y = unify("Salesforce/blip-vqa-base")
y.load()
output = y.inference(payload)

print(output)
```
### **Example of Image Only task**
- This is an example of image only task with the use of "*facebook/sam-vit-base*"
```python
from univlm.Model import unify  
from PIL import Image
import requests

img_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

payload = {"pixel_values": image, "text": None}

y = unify("facebook/sam-vit-base")
y.load()
output = y.inference(payload)
print(output)
```
### **VLLM example**
- This is an example of the use of "*facebook/opt-125m*"
```python
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

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

payload = {"pixel_values": image, "text": None}

y = unify("LiheYoung/depth-anything-large-hf")
y.load()
output = y.inference(payload)
print(output)
```
### **Object detection example**
```python
from univlm.Model import unify  
from PIL import Image
import requests

img_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

payload = {"pixel_values": image, "text": None}

y = unify("hustvl/yolos-tiny")
y.load()
output = y.inference(payload)
print(output)
```


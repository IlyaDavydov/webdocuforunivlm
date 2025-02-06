# Usage Examples

### **Example: Hugging Face Model (Not VLLM)**
- This example demonstrates how to load a Hugging Face model using `Yggdrasil` and perform inference.
```python
y = Yggdrasil("nlptown/bert-base-multilingual-uncased-sentiment",Feature_extractor=False,Image_processor=False,Config_Name="BertForNextSentencePrediction")
y.load()
payload = { "text": "Hello, how are you?", "pixel_values": None }
output = y.inference(payload)
print(output)
```
### **Example of VLM**
img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
listy = [raw_image,raw_image]
payload = {"pixel_values": listy, "text": ["how many dogs?","color of dog"]}

y = Yggdrasil("Salesforce/blip-vqa-base", Feature_extractor=False, Image_processor=False)
y.load()
```
### **Example of Image Only task**
```
img_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

payload = {"pixel_values": image, "text": None}

y = Yggdrasil("facebook/sam-vit-base", Feature_extractor=False, Image_processor=True)
y.load()
output = y.inference(payload)
print(output)

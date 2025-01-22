from transformers import AutoModelForImageClassification
import torch
from flask import Flask, request, jsonify
from PIL import Image
import io
import torchvision.transforms as transforms
import base64


model = AutoModelForImageClassification.from_pretrained("./resnet-50_v1.5")
model.eval()
app = Flask(__name__)

pre_process = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@app.route('/inference', methods=['POST'])
def inference():
    img_file = base64.b64decode(request.json['image'])
    image = Image.open(io.BytesIO(img_file)).convert('RGB')
    image = pre_process(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        logits = output.logits
        predicted_label = logits.argmax(-1).item()
        return jsonify(model.config.id2label[predicted_label])

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)

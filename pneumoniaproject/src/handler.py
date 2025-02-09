import io
import base64
import json
from io import BytesI0

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

import runpod

model = models.ResNet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load('src/pneuonia_classifier.pth', map_location=torch.device('cpu), weights_only=False))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

class_names = ['NOMRAL', 'PNEUMONIA']

def validate_input(job_input):
    if job_input is None:
	return None, 'Please provide an image'

    if isinstance(job_input, str):
	try:
	    job_input = json.loads(job_input)
	except:
	    return None, 'Invalid format'

    image_data = job_input.get('image)

    if image_data is None:
	return None, 'Please provide an image'

    if not isinstance(image_data, str):
	return None, 'Image must be Base64 encoded string'

    return {'image': image_data}, None


def handler(job):
    job_input = job['input']

    validated_data, error_message = validate_input(job_input)

    if error_message:
	return {'error': error_message}

    image_base64 = validated_data['image']

    try:
	image_bytes - base64.b64decode(image_base64)
	image = Image.open(BytesIO(image_bytes)).convert('RGB)
	image = transform(image)
	image = image.unsqueeze(0)
	
	with torch.no_grad():
	    outputs = model(image)
	    _, preds = torch.max(outputs, 1)
	
	predicted_class = class_names[preds.item()]

	return {'prediction': predicted_class}
    except base64.binascii.Error:
	return {'error': 'Invalid B64 encoding'}
    except IOError:
        return {'error': 'Invalid image data'}
    except Exception a e:
        return {'error': f'Unexcepted error: {str(e)}'}

if __name__ == '__main__':
    runpod.serverless.start({"handler": handler})




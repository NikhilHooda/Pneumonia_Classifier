import os
import base64  
import time

import requests
from flask import Flask, render_template, request
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env
API_KEY = os.getenv('API_KEY')

if not API_KEY:
    raise ValueError("API_KEY is missing. Please check your .env file.")

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files['image']
        image_data = base64.b64encode(file.read()).decode('utf-8')

        headers = {
            'Authorization': API_KEY
        }

        res = requests.post('https://api.runpod.ai/v2/xs2v8g689doolz/run', 
                json = {'input': {'image': image_data}},
                headers=headers)

        res_id = res.json()['id']

        print(res.json())

        for _ in range(10):
            status_res = requests.post(f'https://api.runpod.ai/v2/xs2v8g689doolz/status/{res_id}', headers=headers)
            status = status_res.json()
            print(f"API Status: {status}")  # Debugging line

            if status.get('status', '').lower() == 'completed':  # Prevents AttributeError
                prediction = status['output']['prediction']
                print(f"Full API Response: {status}")  
                break
                
            time.sleep(3)
        
            return render_template('index.html', 
                                original_image = f'data:image/jpeg;base64,{image_data}', 
                                prediction = prediction)
            
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

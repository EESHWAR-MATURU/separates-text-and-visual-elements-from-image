import cv2
import imutils
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import requests
import time
from base64 import b64encode
from IPython.display import Image as IPImage
from pylab import rcParams

rcParams['figure.figsize'] = 10, 20

def makeImageData(imgpath):
    with open(imgpath, 'rb') as f:
        ctxt = b64encode(f.read()).decode()
        img_req = {
            'image': {
                'content': ctxt
            },
            'features': [{
                'type': 'DOCUMENT_TEXT_DETECTION',
                'maxResults': 1
            }]
        }
    return json.dumps({"requests": img_req}).encode()

def requestOCR(url, api_key, imgpath):
    imgdata = makeImageData(imgpath)
    response = requests.post(url, 
                             data=imgdata, 
                             params={'key': api_key}, 
                             headers={'Content-Type': 'application/json'})
    return response

def load_api_key(file_path='vision api.json'):
    with open(file_path) as f:
        data = json.load(f)
    return data["api_key"]

ENDPOINT_URL = 'https://vision.googleapis.com/v1/images:annotate'
api_key = load_api_key()

def extract_text_from_image(image_path):
    response = requestOCR(ENDPOINT_URL, api_key, image_path)
    if response.status_code != 200 or response.json().get('error'):
        raise Exception("Error in OCR request")
    text_annotations = response.json()['responses'][0]['textAnnotations']
    return text_annotations

def get_text_blocks(text_annotations):
    text_blocks = []
    for annotation in text_annotations:
        text = annotation["description"]
        vertices = annotation['boundingPoly']['vertices']
        x_min = min(vertex.get('x', 0) for vertex in vertices)
        y_min = min(vertex.get('y', 0) for vertex in vertices)
        x_max = max(vertex.get('x', 0) for vertex in vertices)
        y_max = max(vertex.get('y', 0) for vertex in vertices)
        text_blocks.append({'text': text, 'x_min': x_min, 'y_min': y_min, 'x_max': x_max, 'y_max': y_max})
    return text_blocks

def segment_visual_elements(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    elements = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        element = image[y:y+h, x:x+w]
        elements.append(element)
    return elements

def save_visual_elements(elements, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    paths = []
    for i, element in enumerate(elements):
        path = os.path.join(output_dir, f'element_{i}.png')
        cv2.imwrite(path, element)
        paths.append(path)
    return paths

def generate_html(text_blocks, image_paths):
    html_content = '<html><body>\n'
    for block in text_blocks:
        html_content += f'<p>{block["text"]}</p>\n'
    for path in image_paths:
        html_content += f'<img src="{path}" />\n'
    html_content += '</body></html>'
    return html_content

def main(image_path, output_dir):
    text_annotations = extract_text_from_image(image_path)
    text_blocks = get_text_blocks(text_annotations)
    visual_elements = segment_visual_elements(image_path)
    image_paths = save_visual_elements(visual_elements, output_dir)
    html_content = generate_html(text_blocks, image_paths)
    with open(os.path.join(output_dir, 'output.html'), 'w') as f:
        f.write(html_content)

if __name__ == '__main__':
    image_path = 'image.jpg'  # Path to your input image
    output_dir = 'output.html'  # Output directory
    main(image_path, output_dir)
    IPImage(filename=os.path.join(output_dir, 'output.html'))


from random import randint
from uuid import uuid4
import requests
from PIL import Image
from io import StringIO, BytesIO
import os
from PIL import Image
import base64
import io
import json
import cv2
import numpy as np
import re

def segmap_to_real(seg_map, style="1"):
    """
    Give a "seg map", calls the API and converts it to a realistic image
    """
    urls = ['http://54.191.227.231:443/', 'http://34.221.84.127:443/', 'http://34.216.59.35:443/']
    request_url = urls[randint(0, 2)] + 'nvidia_gaugan_submit_map'
    unique_request = str(uuid4())[:8]
    data = {'imageBase64':seg_map, 'name':unique_request}
    try:
        result = requests.post(request_url, data = data, timeout=2)
    except:
        return (Image.new('RGB', (512, 512)), False)
    if result.status_code != 200:
        print("Error submitting image to GauGan API!")
        return (Image.new('RGB', (512, 512)), False)
    data = {}
    request_url = request_url.replace("nvidia_gaugan_submit_map", "nvidia_gaugan_receive_image")    
    try:
        r = requests.post(request_url, data = {'name':unique_request, "style_name":style}, timeout=5)    
    except:
        return (Image.new('RGB', (512, 512)), False)
    if r.status_code != 200:
        print("Error submitting image to GauGan API!")
        return (Image.new('RGB', (512, 512)), False)
    bytes_data = BytesIO(r.content)
    img = Image.open(bytes_data)
    return (img, True)

def img_to_bytes(image, intermediate="target_images"):
    #path_to_try = f"{os.getcwd()}/{intermediate}/{path}"
    try:
        #image = Image.open(path_to_try)
        imgByteArr = io.BytesIO()
        image.save(imgByteArr, format='PNG')        
        return 'data:image/png;base64,'+ base64.b64encode(imgByteArr.getvalue()).decode('ascii')
    except:
        print(path_to_try)
        return ""
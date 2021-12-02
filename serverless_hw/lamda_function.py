import tflite_runtime.interpreter as tflite
from io import BytesIO
from urllib import request
from PIL import Image
import ssl
import numpy as np

#interpreter = tflite.Interpreter(model_path='cats-dogs.tflite')
interpreter = tflite.Interpreter(model_path='./cats-dogs-v2.tflite')
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

ssl._create_default_https_context = ssl._create_unverified_context


#image_url = "https://upload.wikimedia.org/wikipedia/commons/9/9a/Pug_600.jpg"


def pre_process_input(X):
   return X/255

classes = [
    'dog',
    'cat',
]

def predict(url):
    img = download_image(url)
    img_processed = prepare_image(img,target_size=(150,150))
    x = np.array(img_processed,dtype='float32')
    X = np.array([x])
    X = pre_process_input(X)

    interpreter.set_tensor(input_index,X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    float_pred = float(preds[0,0])
    return float_pred

def lamda_handler(event,context=None):
    url = event['url']
    pred = predict(url)
    result = {'prediction for Dog': pred}
    return result





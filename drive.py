import argparse
import base64
import json
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
import matplotlib.pyplot as plt
import scipy
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None
image_counter=0
@sio.on('telemetry')
def telemetry(sid, data):
    global image_counter
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"] 

    #converting image bytes to image
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    #image to numpy array
    image_array = np.asarray(image)

    #Enable these three lines only if you want to save image
    #scipy.misc.imsave('./output\output_{}.jpg'.format(image_counter), image_array)
    #image_counter+=1

    #Cropping and reducing image size
    image_array=scipy.misc.imresize(image_array[50:120,0:320,:],(100,220,3))

    #reshape image to pass to the model for prediction
    transformed_image_array = image_array.reshape(1,100,220,3)
    #predicting steering angle based on the center camera image from car
    steering_angle =float(model.predict(transformed_image_array, batch_size=1)[0][0])

    #Controlling speed using throttle
    min_speed=16
    max_speed=20
    if(float(speed)<min_speed):
        throttle=1.0
    elif (float(speed)>max_speed):
        throttle=-1.0
    else:
        throttle = 0.2

    print('\033[94m steering:\t\033[92m{:.4f},\t \033[94m throttle:\t\033[92m{:.2f}'.format(steering_angle, throttle))
    #Sending steering and throttle to the simulator using socket
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
                        help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    import matplotlib
    matplotlib.rcParams['interactive'] == True


    #Loading model using Keras
    from keras.models import load_model
    model = load_model(args.model)


    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
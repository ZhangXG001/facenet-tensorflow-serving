from __future__ import print_function
import os
import argparse
import time
import numpy as np
from scipy.misc import imread
from tensorflow.python.framework.tensor_util import MakeNdarray
import grpc
from tensorflow.contrib.util import make_tensor_proto
import h5py
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


def run(host, port, image, model, signature_name):
    # Read an image
    image_dir = image
    img_list = []
    class_name = []
    for file in os.listdir(image_dir):
        img = imread(os.path.join(image_dir, file))
        img = img.astype(np.float32)
        cla = file.split('.')[0]
        class_name.append(cla)
        img_list.append(img)
    images = np.stack(img_list)
    start = time.time()
    # Call prediction model to make prediction on the image
    channel = grpc.insecure_channel('{host}:{port}'.format(host=host, port=port))
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model
    request.model_spec.signature_name = signature_name
    request.inputs['images'].CopyFrom(make_tensor_proto(images, shape=[images.shape[0], 160, 160, 3]))
    result = stub.Predict(request, 10.0)
    out = MakeNdarray(result.outputs['embeddings'])
    end = time.time()
    time_diff = end - start
    #print('out',out)
    print('time elapased: {}'.format(time_diff))
    # Reference:
    # How to access nested values
    # https://stackoverflow.com/questions/44785847/how-to-retrieve-float-val-from-a-predictresponse-object
    start = time.time()
    emb = out
    # emb = np.asarray(emb).squeeze()
    f = h5py.File(r'C:\Users\zxg\Desktop\facenettest\embeddings1.h5py', 'a')
    class_arr = [i for i in class_name]
    print('class_arr',class_arr)
    for i in range(len(class_arr)):
        f.create_dataset(class_arr[i], data=emb[i])
       # print('姓名：', class_arr[i])
        #print('特征：', emb[i])
    f.close()
    end = time.time() - start
    print('time:', end)
    return MakeNdarray(result.outputs['embeddings'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', help='Tensorflow server host name', default='172.20.15.85', type=str)
    parser.add_argument('--port', help='Tensorflow server port number', default=8500, type=int)
    parser.add_argument('--image', help='input image',default=r'C:\Users\zxg\Desktop\facenettest\faceforregister' ,type=str)
    parser.add_argument('--model', help='model name',default='3' ,type=str)
    parser.add_argument('--signature_name', help='Signature name of saved TF model',
                        default='calculate_embeddings', type=str)

    args = parser.parse_args()
run(args.host, args.port, args.image, args.model, args.signature_name)
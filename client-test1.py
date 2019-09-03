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
    out=MakeNdarray(result.outputs['embeddings'])
    end = time.time()
    time_diff = end - start
    print('time elapased: {}'.format(time_diff))


    start = time.time()
    f = h5py.File(r'C:\Users\zxg\Desktop\facenettest\embeddings1.h5py', 'r')
    class_arr = [k for k in f.keys()]
    emb_arr = [f[k].value for k in f.keys()]
    print('emb_arr',emb_arr)
    emb = out
    # emb = np.asarray(emb).squeeze()
    for i in range(len(emb)):
        # print('特征',emb[i])
        diff = np.sum(np.square(emb[i] - emb_arr), axis=1)
        print(diff)
        min_diff = min(diff)
        print('距离:', min_diff)
        if min_diff < 1.24:
            index = np.argmin(diff)
            face_class = class_arr[index]
            print('识别结果:', face_class)
        else:
            print('不能识别')
    end = time.time() - start
    print('识别时间:', end)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', help='Tensorflow server host name', default='172.20.15.85', type=str)
    parser.add_argument('--port', help='Tensorflow server port number', default=8500, type=int)
    parser.add_argument('--image', help='input image',default=r'C:\Users\zxg\Desktop\facenettest\facefortest' ,type=str)
    parser.add_argument('--model', help='model name',default='3' ,type=str)
    parser.add_argument('--signature_name', help='Signature name of saved TF model',
                        default='calculate_embeddings', type=str)

    args = parser.parse_args()
run(args.host, args.port, args.image, args.model, args.signature_name)
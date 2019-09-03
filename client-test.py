
from __future__ import print_function

import time
import os
import sys
import argparse
import h5py
import time
import numpy as np
from scipy import misc
from io import BytesIO
from PIL import Image

from grpc.beta import implementations
from tensorflow.python.framework.tensor_util import MakeNdarray
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import types_pb2

def generate_input_string(image):
    image_data_arr = []
    for i in range(image.shape[0]):
        byte_io = BytesIO()
        img = Image.fromarray(image[i, :, :, :].astype(np.uint8).squeeze())

        img.save(byte_io, 'JPEG')
        byte_io.seek(0)
        image_data = byte_io.read()
        image_data_arr.append([image_data])
    return image_data_arr

def facenet_serving(images):
    host = '172.20.15.85'
    port = 8500

    channel = implementations.insecure_channel(host, port)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    request = predict_pb2.PredictRequest()

    request.model_spec.name = '1'

    request.model_spec.signature_name = 'calculate_embeddings'

    image_data_arr = np.asarray(images)
    #print('image_data_arr',image_data_arr)
    #print('image_data',image_data_arr.dtype)
    input_size = image_data_arr.shape[0]

    dims = [tensor_shape_pb2.TensorShapeProto.Dim(size=input_size)]
    print('dims',dims)
    tensor_shape_proto = tensor_shape_pb2.TensorShapeProto(dim=dims)

    tensor_proto = tensor_pb2.TensorProto(
        dtype=types_pb2.DT_STRING,
        tensor_shape=tensor_shape_proto,
        string_val=[image_data for image_data in image_data_arr])
    print('tensor_proto', tensor_proto)
    request.inputs['images'].CopyFrom(tensor_proto)
    result = stub.Predict(request, 10.0)
    print(MakeNdarray(result.outputs['embeddings']))
    return MakeNdarray(result.outputs['embeddings'])

def main(args):
    start=time.time()
    f = h5py.File(r'C:\Users\zxg\Desktop\facenettest\embeddings.h5py', 'r')
    class_arr = [k for k in f.keys()]
    emb_arr = [f[k].value for k in f.keys()]
    #print(class_arr)
    #print(emb_arr)
    image_dir = args.image_dir
    img_list = []
    for file in os.listdir(image_dir):
        with open(os.path.join(image_dir, file), 'rb') as f:
            img = f.read()
            #img_list =img
            img_list.append(img)
    #images = np.stack(img_list)
    images = img_list

    print('images',images)
    emb = facenet_serving(images)
    #emb = np.asarray(emb).squeeze()
    for i in range(len(emb)):
        #print('特征',emb[i])
        diff = np.sum(np.square(emb[i]- emb_arr), axis=1)
        print(diff)
        min_diff = min(diff)
        print(min_diff)
        print('距离:', min_diff)
        if min_diff < 1.24:
           index = np.argmin(diff)
           face_class = class_arr[index]
           print('识别结果:', face_class)
        else:
           print('不能识别')
    end = time.time()-start
    print('识别时间:',end)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_dir', type=str, help='Image directory.',default=r'C:\Users\zxg\Desktop\facenettest\facefortest')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))



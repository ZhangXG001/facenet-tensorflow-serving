
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


def facenet_serving(images):
    host = '172.20.15.85'
    port = 8500
    channel = implementations.insecure_channel(host, port)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    request = predict_pb2.PredictRequest()

    request.model_spec.name = '1'

    request.model_spec.signature_name = 'calculate_embeddings'

    image_data_arr = np.asarray(images)

    input_size = image_data_arr.shape[0]

    dims = [tensor_shape_pb2.TensorShapeProto.Dim(size=input_size)]
    tensor_shape_proto = tensor_shape_pb2.TensorShapeProto(dim=dims)
    tensor_proto = tensor_pb2.TensorProto(
        dtype=types_pb2.DT_STRING,
        tensor_shape=tensor_shape_proto,
        string_val=[image_data for image_data in image_data_arr])
    request.inputs['images'].CopyFrom(tensor_proto)
    result = stub.Predict(request, 10.0)
    return MakeNdarray(result.outputs['embeddings'])

def main(args):
    start=time.time()
    image_dir = args.image_dir
    img_list = []
    class_name = []
    for file in os.listdir(image_dir):
        with open(os.path.join(image_dir, file), 'rb') as f:
            img = f.read()
            cla = file.split('.')[0]
            class_name.append(cla)
            img_list.append(img)
    images = np.stack(img_list)

    emb = facenet_serving(images)
    #emb = np.asarray(emb).squeeze()
    f = h5py.File(r'C:\Users\zxg\Desktop\facenettest\embeddings.h5py', 'a')
    class_arr = [i for i in class_name]
    for i in range(len(class_arr)):
        f.create_dataset(class_arr[i], data=emb[i])
        print('姓名：',class_arr[i])
        print('特征：',emb[i])
    f.close()
    end = time.time()-start
    print('time:',end)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_dir', type=str, help='Image directory.',default=r'C:\Users\zxg\Desktop\facenettest\faceforregister')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))



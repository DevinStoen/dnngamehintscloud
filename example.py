#!/usr/bin/env python2
# Copyright (c) 2015-2017, NVIDIA CORPORATION.  All rights reserved.

"""
Classify an image using individual model files
Use this script as an example to build your own tool
"""

import argparse
import os
import time

from google.protobuf import text_format
import numpy as np
import PIL.Image
import scipy.misc

os.environ['GLOG_minloglevel'] = '2'  # Suppress most caffe output
import caffe  # noqa
from caffe.proto import caffe_pb2  # noqa


def get_net(caffemodel, deploy_file, use_gpu=True):
    """
    Returns an instance of caffe.Net
    Arguments:
    caffemodel -- path to a .caffemodel file
    deploy_file -- path to a .prototxt file
    Keyword arguments:
    use_gpu -- if True, use the GPU for inference
    """
    if use_gpu:

        caffe.set_mode_gpu()
        #caffe.set_device(0);

    # load a new model
    return caffe.Net(deploy_file, caffemodel, caffe.TEST)


def get_transformer(deploy_file, mean_file=None):
    """
    Returns an instance of caffe.io.Transformer
    Arguments:
    deploy_file -- path to a .prototxt file
    Keyword arguments:
    mean_file -- path to a .binaryproto file (optional)
    """
    network = caffe_pb2.NetParameter()
    with open(deploy_file) as infile:
        text_format.Merge(infile.read(), network)

    if network.input_shape:
        dims = network.input_shape[0].dim
    else:
        dims = network.input_dim[:4]

    t = caffe.io.Transformer(inputs={'data': dims})
    t.set_transpose('data', (2, 0, 1))  # transpose to (channels, height, width)

    # color images
    if dims[1] == 3:
        # channel swap
        t.set_channel_swap('data', (2, 1, 0))

    if mean_file:
        # set mean pixel
        with open(mean_file, 'rb') as infile:
            blob = caffe_pb2.BlobProto()
            blob.MergeFromString(infile.read())
            if blob.HasField('shape'):
                blob_dims = blob.shape
                assert len(blob_dims) == 4, 'Shape should have 4 dimensions - shape is "%s"' % blob.shape
            elif blob.HasField('num') and blob.HasField('channels') and \
                    blob.HasField('height') and blob.HasField('width'):
                blob_dims = (blob.num, blob.channels, blob.height, blob.width)
            else:
                raise ValueError('blob does not provide shape or 4d dimensions')
            pixel = np.reshape(blob.data, blob_dims[1:]).mean(1).mean(1)
            t.set_mean('data', pixel)

    return t


def load_image(path, height, width, mode='RGB'):
    """
    Load an image from disk
    Returns an np.ndarray (channels x width x height)
    Arguments:classify
    path -- path to an image on disk
    width -- resize dimension
    height -- resize dimension
    Keyword arguments:
    mode -- the PIL mode that the image should be converted to
        (RGB for color or L for grayscale)
    """
    image = PIL.Image.open(path)
    image = image.convert(mode)
    image = np.array(image)
    # squash
    image = scipy.misc.imresize(image, (height, width), 'bilinear')
    return image


def forward_pass(images, net, transformer, batch_size=None):
    """
    Returns scores for each image as an np.ndarray (nImages x nClasses)
    Arguments:
    images -- a list of np.ndarrays
    net -- a caffe.Net
    transformer -- a caffe.io.Transformer
    Keyword arguments:
    batch_size -- how many images can be processed at once
        (a high value may result in out-of-memory errors)
    """
    if batch_size is None:
        batch_size = 1

    caffe_images = []
    for image in images:
        if image.ndim == 2:
            caffe_images.append(image[:, :, np.newaxis])
        else:
            caffe_images.append(image)

    dims = transformer.inputs['data'][1:]

    scores = None

    print("print out batch size\n")###
    print(batch_size)##
    print(len(caffe_images))###

    for chunk in [caffe_images[x:x + batch_size] for x in xrange(0, len(caffe_images), batch_size)]:
        new_shape = (len(chunk),) + tuple(dims)
        if net.blobs['data'].data.shape != new_shape:
            net.blobs['data'].reshape(*new_shape)
        for index, image in enumerate(chunk):
            image_data = transformer.preprocess('data', image)
            net.blobs['data'].data[index] = image_data
        start = time.time()
        output = net.forward()[net.outputs[-1]]
        end = time.time()
        if scores is None:
            scores = np.copy(output)
        else:
            scores = np.vstack((scores, output))
        print ('Processed %s/%s images in %f seconds ...' % (len(scores), len(caffe_images), (end - start)))

    return scores


def read_labels(labels_file):
    """
    Returns a list of strings
    Arguments:
    labels_file -- path to a .txt file
    """
    if not labels_file:
        print ('WARNING: No labels file provided. Results will be difficult to interpret.')
        return None

    labels = []
    with open(labels_file) as infile:
        for line in infile:
            label = line.strip()
            if label:
                labels.append(label)
    assert len(labels), 'No labels found'
    return labels


def classify(target, folderPath, textPath, caffemodel, deploy_file, image_files,
             mean_file=None, labels_file=None, batch_size=None, use_gpu=True):
    """F
    Classify some images against a Caffe model and print the results
    Arguments:
    caffemodel -- path to a .caffemodel
    deploy_file -- path to a .prototxt
    image_files -- list of paths to images
    Keyword arguments:
    mean_file -- path to a .binaryproto
    labels_file path to a .txt file
    use_gpu -- if True, run inference on the GPU
    """
    # Load the model and images
    net = get_net(caffemodel, deploy_file, use_gpu)
    transformer = get_transformer(deploy_file, mean_file)
    _, channels, height, width = transformer.inputs['data']
    if channels == 3:
        mode = 'RGB'
    elif channels == 1:
        mode = 'L'
    else:
        raise ValueError('Invalid number for channels: %s' % channels)


    print("Print image files in example.py not working \n")
    print (image_files)## 
    #images = [load_image(image_file, height, width, mode) for image_file in image_files]
    images = image_files
    labels = read_labels(labels_file)

    # Classify the image
    scores = forward_pass(images, net, transformer, batch_size=batch_size)

    #
    # Process the results
    #

    indices = (-scores).argsort()[:, :5]  # take top 5 results
    classifications = []
    for image_index, index_list in enumerate(indices):
        result = []
        for i in index_list:
            # 'i' is a category in labels and also an index into scores
            if labels is None:
                label = 'Class #%s' % i
            else:
                label = labels[i]
            result.append((label, round(100.0 * scores[image_index, i], 4)))
        classifications.append(result)

    
    ##target = open(textPath, "a")
    highestConfidence = 0
    #resultLabel = 'label'
    for index, classification in enumerate(classifications):
        #target.write(' Prediction for %s \n' % image_files[index])
        print ('{:-^80}'.format(' Prediction for %s \n' % image_files[index]))
        for label, confidence in classification:
            print ('{:9.4%} - "{}"\n'.format(confidence / 100.0, label))
            #target.write('%9.4f   %s\n'% (confidence, label))
            if confidence > highestConfidence:
                highestConfidence = confidence
                resultLabel = label
    print ("confidence is " , highestConfidence)####
    #target.write("confidence is %s \n" % highestConfidence)
    print ("Result is " + resultLabel)####
    #target.write("Result is " + resultLabel+"\n")####    

    return resultLabel

#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from nnlib import *
import argparse
import os
import glob

parser = argparse.ArgumentParser()

# Inputs of the model 
parser.add_argument('--path_input', default = './image_predicted_landscape', type = str)
parser.add_argument('--path_output', default = './enhance_image_landscape', type = str)

global args
args = parser.parse_args()

# Inputs of the model 
PATH_INPUT = args.path_input
PATH_OUTPUT = args.path_output


PER_CHANNEL_MEANS = np.array([0.47614917, 0.45001204, 0.40904046])
fns= np.array(glob.glob( os.path.join(PATH_INPUT, '*.jpg') ))
LOAD_SCALE = 1
UPSAMPLING = 4

if not os.path.exists(PATH_OUTPUT):
    os.makedirs(PATH_OUTPUT)

for fn in fns:

    fne = ''.join(fn.split('/')[-1])
    imgs = loadimg(fn, LOAD_SCALE)

    if imgs is None:
        continue
    imgs = np.expand_dims(imgs, axis=0)
    imgsize = np.shape(imgs)[1:]
    print('processing %s' % fn)
    xs = tf.placeholder(tf.float32, [1, imgsize[0], imgsize[1], imgsize[2]])
    rblock = [resi, [[conv], [relu], [conv]]]
    ys_est = NN('generator',
                [xs,
                 [conv], [relu],
                 rblock, rblock, rblock, rblock, rblock,
                 rblock, rblock, rblock, rblock, rblock,
                 [upsample], [conv], [relu],
                 [upsample], [conv], [relu],
                 [conv], [relu],
                 [conv, 3]])
    ys_res = tf.image.resize_images(xs, [UPSAMPLING*imgsize[0], UPSAMPLING*imgsize[1]],
                                    method=tf.image.ResizeMethod.BICUBIC)
    ys_est += ys_res + PER_CHANNEL_MEANS
    sess = tf.InteractiveSession()
    tf.train.Saver().restore(sess, os.getcwd()+'/weights')
    output = sess.run([ys_est, ys_res+PER_CHANNEL_MEANS],
                      feed_dict={xs: imgs-PER_CHANNEL_MEANS})
    saveimg(output[0][0], PATH_OUTPUT + '/' + fne)
    sess.close()
    tf.reset_default_graph()
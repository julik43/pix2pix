from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import os
import time
#import matplotlib.pyplot as plt
import numpy as np
import glob
import scipy.misc
from tensorflow.keras import *
from tensorflow.keras.layers import *
import argparse

import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
	print('no display found. Using non-interactive Agg backend')
	mpl.use('Agg')
import matplotlib.pyplot as plt

def resize(img, height, width):
	img = tf.compat.v1.image.resize_image_with_pad(img, height, width)
	return img

def resize_img(inimg, tgimg, height, width):	
	inimg = resize(inimg, height, width)
	tgimg = resize(tgimg, height, width)
	return inimg, tgimg

def normalize(img):
	img = (img/127.5)-1
	return img

def normalize_img(inimg, tgimg):
	inimg = normalize(inimg)
	tgimg = normalize(tgimg)
	return inimg, tgimg

# data augmentation
@tf.function
def data_augmentation(inimg, tgimg, filename):
	
	inimg, tgimg = resize_img(inimg, tgimg, IMG_HEIGHT + IMG_AUG, IMG_WIDTH + IMG_AUG)
	stacked_image = tf.stack([inimg, tgimg], axis = 0)
	cropped_image = tf.image.random_crop(stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])			
	inimg, tgimg = cropped_image[0], cropped_image[1]

	if tf.random.uniform(()) > 0.5:
		inimg = tf.image.flip_left_right(inimg)
		tgimg = tf.image.flip_left_right(tgimg)

	return inimg, tgimg

@tf.function
def load_image(filename, augment = True):
	inimg = tf.cast(tf.image.decode_jpeg(tf.io.read_file(INPATH + '/' + filename)), tf.float32)[..., :3]
	tgimg = tf.cast(tf.image.decode_jpeg(tf.io.read_file(OUTPATH + '/' + filename)), tf.float32)[..., :3]

	inimg, tgimg = resize_img(inimg, tgimg, IMG_HEIGHT, IMG_WIDTH)

	if augment:
		inimg, tgimg = data_augmentation(inimg, tgimg, filename)

	inimg, tgimg = normalize_img(inimg, tgimg)

	return inimg, tgimg

def load_train_image(filename):
	return load_image(filename, True)

def load_test_image(filename):
	return load_image(filename, False)

@tf.function
def load_pred_image(filename):
	img = tf.cast(tf.image.decode_jpeg(tf.io.read_file(PREDPATH + '/' + filename)), tf.float32)[..., :3]
	img = resize(img, IMG_HEIGHT, IMG_WIDTH)
	img = normalize(img)
	return img

def downsample(filters, size, apply_batchnorm=True):
	initializer = tf.random_normal_initializer(0., 0.02)

	result = Sequential()
	result.add(Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))

	if apply_batchnorm:
		result.add(BatchNormalization())

	result.add(LeakyReLU())

	return result

def upsample(filters, size, apply_dropout=False):
	initializer = tf.random_normal_initializer(0., 0.02)

	result = Sequential()
	result.add(Conv2DTranspose(filters, size, strides=2,padding='same',kernel_initializer=initializer,use_bias=False))

	result.add(BatchNormalization())

	if apply_dropout:
		result.add(Dropout(0.5))

	result.add(ReLU())

	return result

def Generator():
	down_stack = [
		downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
		downsample(128, 4), # (bs, 64, 64, 128)
		downsample(256, 4), # (bs, 32, 32, 256)
		downsample(512, 4), # (bs, 16, 16, 512)
		downsample(512, 4), # (bs, 8, 8, 512)
		downsample(512, 4), # (bs, 4, 4, 512)
		downsample(512, 4), # (bs, 2, 2, 512)
		downsample(512, 4), # (bs, 1, 1, 512)
	]

	up_stack = [
		upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
		upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
		upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
		upsample(512, 4), # (bs, 16, 16, 1024)
		upsample(256, 4), # (bs, 32, 32, 512)
		upsample(128, 4), # (bs, 64, 64, 256)
		upsample(64, 4), # (bs, 128, 128, 128)
	]

	initializer = tf.random_normal_initializer(0., 0.02)
	last = Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=2, padding='same', kernel_initializer=initializer, activation='tanh') # (bs, 256, 256, 3)

	concat = Concatenate()

	inputs = Input(shape=[None,None,3])
	x = inputs

	# Downsampling through the model
	skips = []
	for down in down_stack:		
		x = down(x)
		skips.append(x)

	skips = reversed(skips[:-1])

	# Upsampling and establishing the skip connections
	for up, skip in zip(up_stack, skips):
		
		x = up(x)
		x = concat([x, skip])

	x = last(x)

	return Model(inputs=inputs, outputs=x)

def Discriminator():
	initializer = tf.random_normal_initializer(0., 0.02)

	inp = Input(shape=[None, None, 3], name='input_image')
	tar = Input(shape=[None, None, 3], name='target_image')

	x = concatenate([inp, tar]) # (bs, 256, 256, channels*2)

	down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
	down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
	down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

	# Option 2 as tensorflow
	zero_pad1 = ZeroPadding2D()(down3) # (bs, 34, 34, 256)
	conv = Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

	batchnorm1 = BatchNormalization()(conv)

	leaky_relu = LeakyReLU()(batchnorm1)

	zero_pad2 = ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

	last = Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

	return Model(inputs=[inp, tar], outputs=last)

def discriminator_loss(disc_real_output, disc_generated_output):
	real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
	generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
	total_disc_loss = real_loss + generated_loss
	return total_disc_loss

def generator_loss(disc_generated_output, gen_output, target):
	gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
	# mean absolute error
	l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
	total_gen_loss = gan_loss + (LAMBDA * l1_loss)
	return total_gen_loss


def generate_images(model, test_input, tar, save_filename=False, display_imgs=True, epoch = 0):
	# the training=True is intentional here since
	# we want the batch statistics while running the model
	# on the test dataset. If we use training=False, we will get
	# the accumulated statistics learned from the training dataset
	# (which we don't want)
	prediction = model(test_input, training=True)

	if save_filename:
		tf.keras.preprocessing.image.save_img(PATH_OUTPUT + save_filename + '_pred' + '.jpg', prediction[0,...])

		if epoch == 0:
			tf.keras.preprocessing.image.save_img(PATH_OUTPUT + save_filename + '_input' + '.jpg', test_input[0,...])
			tf.keras.preprocessing.image.save_img(PATH_OUTPUT + save_filename + '_ground_truth' + '.jpg', tar[0,...])

@tf.function
def train_step(input_image, target):

	with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
		gen_output = generator(input_image, training=True)

		disc_real_output = discriminator([input_image, target], training=True)
		disc_generated_output = discriminator([input_image, gen_output], training=True)

		gen_loss = generator_loss(disc_generated_output, gen_output, target)
		disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

		generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
		discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

		generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
		discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

		return gen_loss, disc_loss


def fit(train_ds, epochs, test_ds):
	for epoch in range(epochs):
		start = time.time()

		# Train
		for input_image, target in train_ds:
			gen_loss, disc_loss = train_step(input_image, target)

		# in case is desired to know the loss through the epochs, uncomment this
		# print("gen_loss: " + str(gen_loss) + ". disc_loss: " + str(disc_loss))

		# Test on the same image so that the progress of the model can be easily seen.
		if epoch % 1 == 0:
			imgi = 0
			for example_input, example_target in test_ds.take(EVOLUTION_IMAGES):
				generate_images(generator, example_input, example_target, str(imgi) + '_' + str(epoch), display_imgs = True, epoch = epoch)
				imgi+=1

		# saving (checkpoint) the model every 20 epochs
		if (epoch + 1) % 20 == 0:
			checkpoint.save(file_prefix = checkpoint_prefix)

		print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))


def predict(predict_ds):

	img = 0
	for example_input in predict_ds.take(len(pred_urls)):

		print(pred_urls[img])
		prediction = generator(example_input, training=True)
		tf.keras.preprocessing.image.save_img(PRED_OUTPUT + 'pred_' + pred_urls[img], prediction[0,...])
		img+=1



parser = argparse.ArgumentParser()

# Inputs of the model 
parser.add_argument('--inpath', default = "./train_input", type = str)
parser.add_argument('--outpath', default = './train_desired_output', type = str)
parser.add_argument('--path_output', default = "./output_model/", type = str)
parser.add_argument('--predpath', default = './image_to_pred_landscape', type = str)
parser.add_argument('--pred_output', default = "./image_predicted_landscape/", type = str)
parser.add_argument('--flag_train', default = 'False', type = str)
parser.add_argument('--epochs', default = 800, type = int)
parser.add_argument('--checkpoint_dir', default = './checkpoints_landscape/ckpt-21', type = str)
parser.add_argument('--restore', default = 'True', type = str)
parser.add_argument('--evolution_images', default = 10, type = int)
parser.add_argument('--batch_size_train', default = 10, type = int)
parser.add_argument('--batch_size_test', default = 1, type = int)
parser.add_argument('--img_width', default = 512, type = int)
parser.add_argument('--img_height', default = 512, type = int)
parser.add_argument('--img_aug', default = 40, type = int)
parser.add_argument('--output_channels', default = 3, type = int)
parser.add_argument('--_lambda', default = 100, type = int)

global args
args = parser.parse_args()

# Inputs of the model 
INPATH = args.inpath
OUTPATH = args.outpath
PATH_OUTPUT = args.path_output
PREDPATH = args.predpath
PRED_OUTPUT = args.pred_output

if args.flag_train == 'True':	
	FLAG_TRAIN = True
else:
	FLAG_TRAIN = False

EPOCHS = args.epochs
CHECKPOINT_DIR = args.checkpoint_dir

if args.restore == 'True':	
	RESTORE = True
else:
	RESTORE = False

EVOLUTION_IMAGES = args.evolution_images
BATCH_SIZE_TRAIN = args.batch_size_train
BATCH_SIZE_TEST = args.batch_size_test
IMG_WIDTH = args.img_width
IMG_HEIGHT = args.img_height
IMG_AUG = args.img_aug
OUTPUT_CHANNELS = args.output_channels
LAMBDA = args._lambda

if not os.path.exists(INPATH): 
	print(str(INPATH) + " does not exist.")
	exit(0)
if not os.path.exists(OUTPATH): 
	print(str(OUTPATH) + " does not exist.")
	exit(0)
if not os.path.exists(PREDPATH): 
	print(str(PREDPATH) + " does not exist.")
	exit(0)	

if not os.path.exists(PATH_OUTPUT):
	os.mkdir(PATH_OUTPUT)
if not os.path.exists(PRED_OUTPUT):
	os.mkdir(PRED_OUTPUT)

# Finding the path to the images
imgurls = [os.path.basename(x) for x in np.array(glob.glob( os.path.join(INPATH, '*.jpg') ))]

#####################################################################
# # to find if the images have only one channel and delete them
#####################################################################
for x in range(0,len(imgurls)):
	image = plt.imread(INPATH + str('/') +imgurls[x])

	if len(image.shape) <3:
		print(imgurls[x])
		os.remove(INPATH + str('/') +imgurls[x])

	image = plt.imread(OUTPATH + str('/') +imgurls[x])

	if len(image.shape) <3:
		print(imgurls[x])
		os.remove(OUTPATH + str('/') +imgurls[x])


n = len(imgurls)
train_n = round(n*0.80)
randurls = np.copy(imgurls)
np.random.seed()
np.random.shuffle(randurls)

# Partition train/test
tr_urls = randurls[:train_n]
ts_urls = randurls[train_n:n]

# verifying if there are enough images, if not EVOLUTION_IMAGES is set to the maximum
if EVOLUTION_IMAGES > len(ts_urls):
	EVOLUTION_IMAGES = len(ts_urls)

# creating train and test datasets
train_dataset = tf.data.Dataset.from_tensor_slices(tr_urls)
train_dataset = train_dataset.map(load_train_image, num_parallel_calls = 3)
train_dataset = train_dataset.batch(BATCH_SIZE_TRAIN)

test_dataset = tf.data.Dataset.from_tensor_slices(ts_urls)
test_dataset = test_dataset.map(load_test_image, num_parallel_calls = 3)
test_dataset = test_dataset.batch(BATCH_SIZE_TEST)

# creating generator and discriminator for the model
generator = Generator()
discriminator = Discriminator()
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_prefix = os.path.join(CHECKPOINT_DIR, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
	discriminator_optimizer=discriminator_optimizer,
	generator=generator,
	discriminator=discriminator)

# Restore cars
if RESTORE == True:
	# checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_DIR))
	checkpoint.restore(CHECKPOINT_DIR)

if FLAG_TRAIN:
	fit(train_dataset, EPOCHS, test_dataset)
else:
	pred_urls= [os.path.basename(x) for x in np.array(glob.glob( os.path.join(PREDPATH, '*.jpg') ))]
	pred_dataset = tf.data.Dataset.from_tensor_slices(pred_urls)
	pred_dataset = pred_dataset.map(load_pred_image, num_parallel_calls = 1)
	pred_dataset = pred_dataset.batch(1)
	predict(pred_dataset)




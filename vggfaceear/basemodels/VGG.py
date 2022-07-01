import os

## tensorflow version: 2.*
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from tensorflow.keras.layers import Conv2D, BatchNormalization, Lambda, AveragePooling2D, concatenate
from tensorflow.keras import backend as K

import tensorflow_addons as tfa
from tensorflow_addons.losses import TripletSemiHardLoss

ROOT_DIR = os.path.abspath("")

def loadModel(features_from_layer = 1):
	global layer
	layer = features_from_layer

def loadVGG16():
	print("Using VGG16 pretrained model..." )
	
	vgg16 = VGG16(weights="imagenet", 
             include_top=True, 
             input_shape=(224, 224, 3))

	vgg16_descriptor = Model(inputs=vgg16.layers[0].input, outputs=vgg16.layers[-layer].output)

	return vgg16_descriptor

def loadVGG16Face():
	print("Using VGG16 pretrained model adjusted to FACES..." )
	
	model_path = os.path.join(ROOT_DIR, "weights", "vgg16_Face_aug.h5")

	try:
		vgg16face = load_model(model_path)
	except Exception as err:
		print(str(err))
		print("**VGG16-Face model could not be loaded.")
		#ToDo Warning: vgg_face_weights.h5 file should be in weights folder
		
	vgg16face_descriptor = Model(inputs=vgg16face.layers[0].input, outputs=vgg16face.layers[-2].output)

	return vgg16face_descriptor


def loadVGG16Ear():
	print("Using VGG16 pretrained model adjusted to EARS..." )
	
	model_path = os.path.join(ROOT_DIR, "weights", "vgg16_Ear_aug.h5")

	try:
		vgg16ear = load_model(model_path)
	except Exception as err:
		print(str(err))
		print("**VGG16-Ear model could not be loaded.")
		#ToDo Warning: vgg_face_weights.h5 file should be in weights folder
		
	vgg16ear_descriptor = Model(inputs=vgg16ear.layers[0].input, outputs=vgg16ear.layers[-2].output)

	return vgg16ear_descriptor


def loadVGG16Fusion():
	print("Using VGG16 MultiStream for FACES & EARS..." )
	
	model_path = os.path.join(ROOT_DIR, "weights", "vgg16_Face_Ear.h5")

	try:
		vgg16fe = load_model(model_path)
	except Exception as err:
		print(str(err))
		print("**VGG16-Face-Ear model could not be loaded.")
		#ToDo Warning: vgg_face_weights.h5 file should be in weights folder
		
	vgg16fe_descriptor = Model(inputs=[vgg16fe.layers[0].input, vgg16fe.layers[1].input] , outputs=vgg16fe.layers[-3].output)

	return vgg16fe_descriptor

def loadVGGEar():
	print("Using VGGEar (trained on datasets VGGFace and UERC)..." )
	
	model_path = os.path.join(ROOT_DIR, "weights", "vgg16_vggface_plus_uerc.h5")

	try:
		vggear = load_model(model_path)
	except Exception as err:
		print(str(err))
		print("**VGGEar model could not be loaded.")
		#ToDo Warning: vgg_face_weights.h5 file should be in weights folder
		
	vggear_descriptor = Model(inputs=vggear.layers[0].input, outputs=vggear.layers[-7].output)

	return vggear_descriptor

def loadVGGFace():
	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
	model.add(Convolution2D(64, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(Convolution2D(4096, (7, 7), activation='relu'))
	model.add(Dropout(0.5))
	model.add(Convolution2D(4096, (1, 1), activation='relu'))
	model.add(Dropout(0.5))
	model.add(Convolution2D(2622, (1, 1)))
	model.add(Flatten())
	model.add(Activation('softmax'))

	weights = os.path.join(ROOT_DIR, "weights", "vgg_face_weights.h5")

	try:
		model.load_weights(weights)
	except Exception as err:
		print(str(err))
		print("**VGGFace weights could not be loaded.")
		#ToDo Warning: vgg_face_weights.h5 file should be in weights folder
	
	##-> layer -3 -5 -7
	descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-layer-1].output)

	return descriptor


def loadOpenFace2(url = 'https://drive.google.com/uc?id=1LSe1YCV1x-BfNnfb7DFZTNpv_Q9jITxn'):
	myInput = Input(shape=(96, 96, 3))

	x = ZeroPadding2D(padding=(3, 3), input_shape=(96, 96, 3))(myInput)
	x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
	x = BatchNormalization(axis=3, epsilon=0.00001, name='bn1')(x)
	x = Activation('relu')(x)
	x = ZeroPadding2D(padding=(1, 1))(x)
	x = MaxPooling2D(pool_size=3, strides=2)(x)
	x = Lambda(lambda x: tf.nn.lrn(x, alpha=1e-4, beta=0.75), name='lrn_1')(x)
	x = Conv2D(64, (1, 1), name='conv2')(x)
	x = BatchNormalization(axis=3, epsilon=0.00001, name='bn2')(x)
	x = Activation('relu')(x)
	x = ZeroPadding2D(padding=(1, 1))(x)
	x = Conv2D(192, (3, 3), name='conv3')(x)
	x = BatchNormalization(axis=3, epsilon=0.00001, name='bn3')(x)
	x = Activation('relu')(x)
	x = Lambda(lambda x: tf.nn.lrn(x, alpha=1e-4, beta=0.75), name='lrn_2')(x) #x is equal added
	x = ZeroPadding2D(padding=(1, 1))(x)
	x = MaxPooling2D(pool_size=3, strides=2)(x)

	# Inception3a
	inception_3a_3x3 = Conv2D(96, (1, 1), name='inception_3a_3x3_conv1')(x)
	inception_3a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_3x3_bn1')(inception_3a_3x3)
	inception_3a_3x3 = Activation('relu')(inception_3a_3x3)
	inception_3a_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3a_3x3)
	inception_3a_3x3 = Conv2D(128, (3, 3), name='inception_3a_3x3_conv2')(inception_3a_3x3)
	inception_3a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_3x3_bn2')(inception_3a_3x3)
	inception_3a_3x3 = Activation('relu')(inception_3a_3x3)

	inception_3a_5x5 = Conv2D(16, (1, 1), name='inception_3a_5x5_conv1')(x)
	inception_3a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_5x5_bn1')(inception_3a_5x5)
	inception_3a_5x5 = Activation('relu')(inception_3a_5x5)
	inception_3a_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3a_5x5)
	inception_3a_5x5 = Conv2D(32, (5, 5), name='inception_3a_5x5_conv2')(inception_3a_5x5)
	inception_3a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_5x5_bn2')(inception_3a_5x5)
	inception_3a_5x5 = Activation('relu')(inception_3a_5x5)

	inception_3a_pool = MaxPooling2D(pool_size=3, strides=2)(x)
	inception_3a_pool = Conv2D(32, (1, 1), name='inception_3a_pool_conv')(inception_3a_pool)
	inception_3a_pool = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_pool_bn')(inception_3a_pool)
	inception_3a_pool = Activation('relu')(inception_3a_pool)
	inception_3a_pool = ZeroPadding2D(padding=((3, 4), (3, 4)))(inception_3a_pool)

	inception_3a_1x1 = Conv2D(64, (1, 1), name='inception_3a_1x1_conv')(x)
	inception_3a_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_1x1_bn')(inception_3a_1x1)
	inception_3a_1x1 = Activation('relu')(inception_3a_1x1)

	inception_3a = concatenate([inception_3a_3x3, inception_3a_5x5, inception_3a_pool, inception_3a_1x1], axis=3)

	# Inception3b
	inception_3b_3x3 = Conv2D(96, (1, 1), name='inception_3b_3x3_conv1')(inception_3a)
	inception_3b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_3x3_bn1')(inception_3b_3x3)
	inception_3b_3x3 = Activation('relu')(inception_3b_3x3)
	inception_3b_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3b_3x3)
	inception_3b_3x3 = Conv2D(128, (3, 3), name='inception_3b_3x3_conv2')(inception_3b_3x3)
	inception_3b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_3x3_bn2')(inception_3b_3x3)
	inception_3b_3x3 = Activation('relu')(inception_3b_3x3)

	inception_3b_5x5 = Conv2D(32, (1, 1), name='inception_3b_5x5_conv1')(inception_3a)
	inception_3b_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_5x5_bn1')(inception_3b_5x5)
	inception_3b_5x5 = Activation('relu')(inception_3b_5x5)
	inception_3b_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3b_5x5)
	inception_3b_5x5 = Conv2D(64, (5, 5), name='inception_3b_5x5_conv2')(inception_3b_5x5)
	inception_3b_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_5x5_bn2')(inception_3b_5x5)
	inception_3b_5x5 = Activation('relu')(inception_3b_5x5)

	inception_3b_pool = Lambda(lambda x: x**2, name='power2_3b')(inception_3a)
	inception_3b_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_3b_pool)
	inception_3b_pool = Lambda(lambda x: x*9, name='mult9_3b')(inception_3b_pool)
	inception_3b_pool = Lambda(lambda x: K.sqrt(x), name='sqrt_3b')(inception_3b_pool)
	inception_3b_pool = Conv2D(64, (1, 1), name='inception_3b_pool_conv')(inception_3b_pool)
	inception_3b_pool = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_pool_bn')(inception_3b_pool)
	inception_3b_pool = Activation('relu')(inception_3b_pool)
	inception_3b_pool = ZeroPadding2D(padding=(4, 4))(inception_3b_pool)

	inception_3b_1x1 = Conv2D(64, (1, 1), name='inception_3b_1x1_conv')(inception_3a)
	inception_3b_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_1x1_bn')(inception_3b_1x1)
	inception_3b_1x1 = Activation('relu')(inception_3b_1x1)

	inception_3b = concatenate([inception_3b_3x3, inception_3b_5x5, inception_3b_pool, inception_3b_1x1], axis=3)

	# Inception3c
	inception_3c_3x3 = Conv2D(128, (1, 1), strides=(1, 1), name='inception_3c_3x3_conv1')(inception_3b)
	inception_3c_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3c_3x3_bn1')(inception_3c_3x3)
	inception_3c_3x3 = Activation('relu')(inception_3c_3x3)
	inception_3c_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3c_3x3)
	inception_3c_3x3 = Conv2D(256, (3, 3), strides=(2, 2), name='inception_3c_3x3_conv'+'2')(inception_3c_3x3)
	inception_3c_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3c_3x3_bn'+'2')(inception_3c_3x3)
	inception_3c_3x3 = Activation('relu')(inception_3c_3x3)

	inception_3c_5x5 = Conv2D(32, (1, 1), strides=(1, 1), name='inception_3c_5x5_conv1')(inception_3b)
	inception_3c_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3c_5x5_bn1')(inception_3c_5x5)
	inception_3c_5x5 = Activation('relu')(inception_3c_5x5)
	inception_3c_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3c_5x5)
	inception_3c_5x5 = Conv2D(64, (5, 5), strides=(2, 2), name='inception_3c_5x5_conv'+'2')(inception_3c_5x5)
	inception_3c_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3c_5x5_bn'+'2')(inception_3c_5x5)
	inception_3c_5x5 = Activation('relu')(inception_3c_5x5)

	inception_3c_pool = MaxPooling2D(pool_size=3, strides=2)(inception_3b)
	inception_3c_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(inception_3c_pool)

	inception_3c = concatenate([inception_3c_3x3, inception_3c_5x5, inception_3c_pool], axis=3)

	#inception 4a
	inception_4a_3x3 = Conv2D(96, (1, 1), strides=(1, 1), name='inception_4a_3x3_conv'+'1')(inception_3c)
	inception_4a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4a_3x3_bn'+'1')(inception_4a_3x3)
	inception_4a_3x3 = Activation('relu')(inception_4a_3x3)
	inception_4a_3x3 = ZeroPadding2D(padding=(1, 1))(inception_4a_3x3)
	inception_4a_3x3 = Conv2D(192, (3, 3), strides=(1, 1), name='inception_4a_3x3_conv'+'2')(inception_4a_3x3)
	inception_4a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4a_3x3_bn'+'2')(inception_4a_3x3)
	inception_4a_3x3 = Activation('relu')(inception_4a_3x3)

	inception_4a_5x5 = Conv2D(32, (1,1), strides=(1,1), name='inception_4a_5x5_conv1')(inception_3c)
	inception_4a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4a_5x5_bn1')(inception_4a_5x5)
	inception_4a_5x5 = Activation('relu')(inception_4a_5x5)
	inception_4a_5x5 = ZeroPadding2D(padding=(2,2))(inception_4a_5x5)
	inception_4a_5x5 = Conv2D(64, (5,5), strides=(1,1), name='inception_4a_5x5_conv'+'2')(inception_4a_5x5)
	inception_4a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4a_5x5_bn'+'2')(inception_4a_5x5)
	inception_4a_5x5 = Activation('relu')(inception_4a_5x5)

	inception_4a_pool = Lambda(lambda x: x**2, name='power2_4a')(inception_3c)
	inception_4a_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_4a_pool)
	inception_4a_pool = Lambda(lambda x: x*9, name='mult9_4a')(inception_4a_pool)
	inception_4a_pool = Lambda(lambda x: K.sqrt(x), name='sqrt_4a')(inception_4a_pool)

	inception_4a_pool = Conv2D(128, (1,1), strides=(1,1), name='inception_4a_pool_conv'+'')(inception_4a_pool)
	inception_4a_pool = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4a_pool_bn'+'')(inception_4a_pool)
	inception_4a_pool = Activation('relu')(inception_4a_pool)
	inception_4a_pool = ZeroPadding2D(padding=(2, 2))(inception_4a_pool)

	inception_4a_1x1 = Conv2D(256, (1, 1), strides=(1, 1), name='inception_4a_1x1_conv'+'')(inception_3c)
	inception_4a_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4a_1x1_bn'+'')(inception_4a_1x1)
	inception_4a_1x1 = Activation('relu')(inception_4a_1x1)

	inception_4a = concatenate([inception_4a_3x3, inception_4a_5x5, inception_4a_pool, inception_4a_1x1], axis=3)

	#inception4e
	inception_4e_3x3 = Conv2D(160, (1,1), strides=(1,1), name='inception_4e_3x3_conv'+'1')(inception_4a)
	inception_4e_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4e_3x3_bn'+'1')(inception_4e_3x3)
	inception_4e_3x3 = Activation('relu')(inception_4e_3x3)
	inception_4e_3x3 = ZeroPadding2D(padding=(1, 1))(inception_4e_3x3)
	inception_4e_3x3 = Conv2D(256, (3,3), strides=(2,2), name='inception_4e_3x3_conv'+'2')(inception_4e_3x3)
	inception_4e_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4e_3x3_bn'+'2')(inception_4e_3x3)
	inception_4e_3x3 = Activation('relu')(inception_4e_3x3)

	inception_4e_5x5 = Conv2D(64, (1,1), strides=(1,1), name='inception_4e_5x5_conv'+'1')(inception_4a)
	inception_4e_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4e_5x5_bn'+'1')(inception_4e_5x5)
	inception_4e_5x5 = Activation('relu')(inception_4e_5x5)
	inception_4e_5x5 = ZeroPadding2D(padding=(2, 2))(inception_4e_5x5)
	inception_4e_5x5 = Conv2D(128, (5,5), strides=(2,2), name='inception_4e_5x5_conv'+'2')(inception_4e_5x5)
	inception_4e_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4e_5x5_bn'+'2')(inception_4e_5x5)
	inception_4e_5x5 = Activation('relu')(inception_4e_5x5)

	inception_4e_pool = MaxPooling2D(pool_size=3, strides=2)(inception_4a)
	inception_4e_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(inception_4e_pool)

	inception_4e = concatenate([inception_4e_3x3, inception_4e_5x5, inception_4e_pool], axis=3)

	#inception5a
	inception_5a_3x3 = Conv2D(96, (1,1), strides=(1,1), name='inception_5a_3x3_conv'+'1')(inception_4e)
	inception_5a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_5a_3x3_bn'+'1')(inception_5a_3x3)
	inception_5a_3x3 = Activation('relu')(inception_5a_3x3)
	inception_5a_3x3 = ZeroPadding2D(padding=(1, 1))(inception_5a_3x3)
	inception_5a_3x3 = Conv2D(384, (3,3), strides=(1,1), name='inception_5a_3x3_conv'+'2')(inception_5a_3x3)
	inception_5a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_5a_3x3_bn'+'2')(inception_5a_3x3)
	inception_5a_3x3 = Activation('relu')(inception_5a_3x3)

	inception_5a_pool = Lambda(lambda x: x**2, name='power2_5a')(inception_4e)
	inception_5a_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_5a_pool)
	inception_5a_pool = Lambda(lambda x: x*9, name='mult9_5a')(inception_5a_pool)
	inception_5a_pool = Lambda(lambda x: K.sqrt(x), name='sqrt_5a')(inception_5a_pool)

	inception_5a_pool = Conv2D(96, (1,1), strides=(1,1), name='inception_5a_pool_conv'+'')(inception_5a_pool)
	inception_5a_pool = BatchNormalization(axis=3, epsilon=0.00001, name='inception_5a_pool_bn'+'')(inception_5a_pool)
	inception_5a_pool = Activation('relu')(inception_5a_pool)
	inception_5a_pool = ZeroPadding2D(padding=(1,1))(inception_5a_pool)

	inception_5a_1x1 = Conv2D(256, (1,1), strides=(1,1), name='inception_5a_1x1_conv'+'')(inception_4e)
	inception_5a_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_5a_1x1_bn'+'')(inception_5a_1x1)
	inception_5a_1x1 = Activation('relu')(inception_5a_1x1)

	inception_5a = concatenate([inception_5a_3x3, inception_5a_pool, inception_5a_1x1], axis=3)

	#inception_5b
	inception_5b_3x3 = Conv2D(96, (1,1), strides=(1,1), name='inception_5b_3x3_conv'+'1')(inception_5a)
	inception_5b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_5b_3x3_bn'+'1')(inception_5b_3x3)
	inception_5b_3x3 = Activation('relu')(inception_5b_3x3)
	inception_5b_3x3 = ZeroPadding2D(padding=(1,1))(inception_5b_3x3)
	inception_5b_3x3 = Conv2D(384, (3,3), strides=(1,1), name='inception_5b_3x3_conv'+'2')(inception_5b_3x3)
	inception_5b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_5b_3x3_bn'+'2')(inception_5b_3x3)
	inception_5b_3x3 = Activation('relu')(inception_5b_3x3)

	inception_5b_pool = MaxPooling2D(pool_size=3, strides=2)(inception_5a)

	inception_5b_pool = Conv2D(96, (1,1), strides=(1,1), name='inception_5b_pool_conv'+'')(inception_5b_pool)
	inception_5b_pool = BatchNormalization(axis=3, epsilon=0.00001, name='inception_5b_pool_bn'+'')(inception_5b_pool)
	inception_5b_pool = Activation('relu')(inception_5b_pool)

	inception_5b_pool = ZeroPadding2D(padding=(1, 1))(inception_5b_pool)

	inception_5b_1x1 = Conv2D(256, (1,1), strides=(1,1), name='inception_5b_1x1_conv'+'')(inception_5a)
	inception_5b_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_5b_1x1_bn'+'')(inception_5b_1x1)
	inception_5b_1x1 = Activation('relu')(inception_5b_1x1)

	inception_5b = concatenate([inception_5b_3x3, inception_5b_pool, inception_5b_1x1], axis=3)

	av_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1))(inception_5b)
	reshape_layer = Flatten()(av_pool)
	dense_layer = Dense(128, name='dense_layer')(reshape_layer)
	norm_layer = Lambda(lambda  x: K.l2_normalize(x, axis=1), name='norm_layer')(dense_layer)

	# Final Model
	model = Model(inputs=[myInput], outputs=norm_layer)
	
	weights = os.path.join(ROOT_DIR, "weights", "openface_weights.h5")

	try:
		model.load_weights(weights)
	except Exception as err:
		print(str(err))
		print("**OpenFace weights could not be loaded.")
		#ToDo Warning: openface_weights.h5 file should be in weights folder
	
	return model


def loadVGGFace_Face():
	print("Using VGGFace pretrained model adjusted to 25 class FACES..." )
	
	model_path = os.path.join(ROOT_DIR, "weights", "vggFace_25class_aug.h5")

	try:
		vggface = load_model(model_path)
	except Exception as err:
		print(str(err))
		print("**VGGFace model could not be loaded.")
		#ToDo Warning: vgg_face_weights.h5 file should be in weights folder
		
	vggface_descriptor = Model(inputs=vggface.layers[0].input, outputs=vggface.layers[-2].output)

	return vggface_descriptor


def loadVGGFace_Ear():
	print("Using VGGFace pretrained model adjusted to 25 class EARS..." )
	
	model_path = os.path.join(ROOT_DIR, "weights", "vggEar_25class_aug.h5")

	try:
		vggear = load_model(model_path)
	except Exception as err:
		print(str(err))
		print("**VGGFace model could not be loaded.")
		#ToDo Warning: vgg_face_weights.h5 file should be in weights folder
		
	vggear_descriptor = Model(inputs=vggear.layers[0].input, outputs=vggear.layers[-2].output)

	return vggear_descriptor


def loadVGGFace_Fusion():
	print("Using VGGFace MultiStream for 25 class FACES & EARS..." )
	
	model_path = os.path.join(ROOT_DIR, "weights", "vggFaceEar_25class_aug.h5")

	try:
		vggfe = load_model(model_path)
	except Exception as err:
		print(str(err))
		print("**VGGFace model could not be loaded.")
		#ToDo Warning: vgg_face_weights.h5 file should be in weights folder
		
	vggfe_descriptor = Model(inputs=[vggfe.layers[0].input, vggfe.layers[1].input] , outputs=vggfe.layers[-3].output)

	return vggfe_descriptor


def loadVGGFace_EDA_first():
	print("Using VGGFace EDA1" )
	
	model_path = os.path.join(ROOT_DIR, "weights", "vggFace_EDA1_wput_aug.h5")

	try:
		vggfeda = load_model(model_path)
	except Exception as err:
		print(str(err))
		print("**VGGFace EDA model could not be loaded.")
		#ToDo Warning: vgg_face_weights.h5 file should be in weights folder
		
	vggfeda_descriptor = Model(inputs=vggfeda.layers[0].input , outputs=vggfeda.layers[-2].output)

	return vggfeda_descriptor


def loadVGGFace_EDA_second():
	print("Using VGGFace EDA2" )
	
	model_path = os.path.join(ROOT_DIR, "weights", "vggFace_EDA2_vggfacetrain_aug.h5")

	try:
		vggfeda = load_model(model_path)
	except Exception as err:
		print(str(err))
		print("**VGGFace EDA model could not be loaded.")
		#ToDo Warning: vgg_face_weights.h5 file should be in weights folder
		
	vggfeda_descriptor = Model(inputs=vggfeda.layers[0].input , outputs=vggfeda.layers[-2].output)

	return vggfeda_descriptor


def loadVGGFace_EDA_third():
	print("Using VGGFace EDA3" )
	
	model_path = os.path.join(ROOT_DIR, "exweights", "vggFace_EDA_vggfacetrain_plus_uerctrain_aug_407classes_2round.h5")
	print(model_path)
	try:
		vggfeda = load_model(model_path)
	except Exception as err:
		print(str(err))
		print("**VGGFace EDA3 model could not be loaded.")
		#ToDo Warning: vgg_face_weights.h5 file should be in weights folder
		
	vggfeda_descriptor = Model(inputs=vggfeda.layers[0].input , outputs=vggfeda.layers[-2].output)

	return vggfeda_descriptor


def loadVGGFaceEar_MS():
	print("Using VGGFaceEar MultiStream trained with 101 classes..." )
	
	model_path = os.path.join(ROOT_DIR, "weights", "VGG_MS_VGGFace_VGGFaceEDA3_101class_avglayer.h5")

	try:
		vggfems = load_model(model_path)
	except Exception as err:
		print(str(err))
		print("**VGGFaceEar model could not be loaded.")
		#ToDo Warning: vgg_face_weights.h5 file should be in weights folder
	
	## INPUT: inputs=[model_face.input, model_ear.input]
	vggfems_descriptor = Model(inputs=[vggfems.inputs[0], vggfems.inputs[1]] , outputs=vggfems.layers[-2].output)

	return vggfems_descriptor


### New Experiments CLEI 2021

def loadVGG16_Ear():
	print("Using VGG16 adjusted to Ears train on 450 (VGGFace) classes" )
	
	# vgg16_vggface450class_allclayers.h5
	# vgg16_vggface450class_3groconv.h5

	model_path = os.path.join(ROOT_DIR, "exweights", "vgg16_vggface450class.h5")

	try:
		vggfeda = load_model(model_path)
	except Exception as err:
		print(str(err))
		print("**VGGFace VGG16_Ear model could not be loaded.")
		#ToDo Warning: vgg_face_weights.h5 file should be in weights folder
		
	vggfeda_descriptor = Model(inputs=vggfeda.layers[0].input , outputs=vggfeda.layers[-layer].output)

	return vggfeda_descriptor


def loadVGGFace_Ear_2():
	print("Using VGGFace Adjusted to Ears on 450 (VGGFace) classes ")
	
	## vggFace-Ear_vggface450class.h5 
	## vggFace-Ear_trainUERC_3groconv.h5   ## UERC test 73.33%	80.89%	84.89%	88.06%	89.72%

	model_path = os.path.join(ROOT_DIR, "exweights", "vggFace-Ear_vggface450class.h5")

	try:
		vggfeda = load_model(model_path)
	except Exception as err:
		print(str(err))
		print("**VGGFace model could not be loaded.")
		#ToDo Warning: vgg_face_weights.h5 file should be in weights folder
		
	vggfeda_descriptor = Model(inputs=vggfeda.layers[0].input , outputs=vggfeda.layers[-layer].output)

	return vggfeda_descriptor


### Experiments Model Surgery

def loadVGGFace_Ear_160():
	print("Using VGGFace Adjusted to Ears (Input size 160x160)")
	
	model_path = os.path.join(ROOT_DIR, "weights", ".h5")

	try:
		vggfeda = load_model(model_path)
	except Exception as err:
		print(str(err))
		print("**VGGFace model could not be loaded.")
		#ToDo Warning: vgg_face_weights.h5 file should be in weights folder
		
	vggfeda_descriptor = Model(inputs=vggfeda.layers[0].input , outputs=vggfeda.layers[-2].output)

	return vggfeda_descriptor

### New Experiments Paper Sensors

def loadVGGFace_Ear_2nd():
	print("Using VGGFace Adjusted to Ears (Input size 224x224)")
	
	model_path = os.path.join(ROOT_DIR, "weights", "vggface_dsvggface2_800class_tlfromvggface_alllayers_sgd1e3_224x224.h5")

	try:
		vggfeda = load_model(model_path)
	except Exception as err:
		print(str(err))
		print("**VGGFace model could not be loaded.")
		#ToDo Warning: vgg_face_weights.h5 file should be in weights folder
		
	vggfeda_descriptor = Model(inputs=vggfeda.layers[0].input , outputs=vggfeda.layers[-4].output)

	return vggfeda_descriptor

def loadVGGFace_Ear_3rd(): ## VGGEar
	print("Using VGGFace Adjusted to Ears (Input size 224x224)")
	
	#model_path = os.path.join(ROOT_DIR, "weights", "vggface_dsvggface2_tf_from800_to_600class_3convgroups_sgd1e4.h5")	
	model_path = os.path.join(ROOT_DIR, "weights", "sensors_test", "vggface_dsvggface2_tf_from800_to_600class_3convgroups_sgd1e5_1.h5")

	try:
		vggfeda = load_model(model_path)
	except Exception as err:
		print(str(err))
		print("**VGGFace model could not be loaded.")
		#ToDo Warning: vgg_face_weights.h5 file should be in weights folder
	##-> layer -1 -2 -4
	vggfeda_descriptor = Model(inputs=vggfeda.layers[0].input , outputs=vggfeda.layers[-layer].output)

	return vggfeda_descriptor

def loadVGGFace_Ear_3rd_uerc():
	print("Using VGGFace3 Adjusted to Ears of UERC dataset (Input size 224x224)")
	
	model_path = os.path.join(ROOT_DIR, "weights", "sensors_test", "vggface_dsvggface2_from600_to_166class_3convgroups_sgd1e5_3.h5")

	try:
		vggfeda = load_model(model_path)
	except Exception as err:
		print(str(err))
		print("**VGGFace model could not be loaded.")
		#ToDo Warning: vgg_face_weights.h5 file should be in weights folder
		
	vggfeda_descriptor = Model(inputs=vggfeda.layers[0].input , outputs=vggfeda.layers[-2].output)

	return vggfeda_descriptor

def loadVGGFace_Ear_3rd_earvn():
	print("Using VGGFace3 Adjusted to Ears of EarVN10 dataset (Input size 224x224)")
	
	#vggface_dsvggface2_from600_to_106class_2convgroups_sgd1e4.h5
	model_path = os.path.join(ROOT_DIR, "weights", "test", "vggface_dsearvn1_from600_to_106class_3convgroups_sgd1e5_attemp7.h5")

	try:
		vggfeda = load_model(model_path)
	except Exception as err:
		print(str(err))
		print("**VGGFace model could not be loaded.")
		#ToDo Warning: vgg_face_weights.h5 file should be in weights folder
		
	vggfeda_descriptor = Model(inputs=vggfeda.layers[0].input , outputs=vggfeda.layers[-6].output)

	return vggfeda_descriptor


def loadVGG16_Ear_CustomInput():
	print("Using VGG16 Adjusted to Ears (Input size 224X112)")
	
	#vgg16_dsvggface2_600class_alllayers_sgd1e4_attempt4_224x112
	#vgg16_dsvggface2_600class_alllayers_sgd1e4_attempt5_2_224x112.h5
	model_path = os.path.join(ROOT_DIR, "weights", "sensors_test", "vgg16_dsvggface2_600class_alllayers_sgd1e4_attempt5_2_224x112.h5")

	try:
		vggfeda = load_model(model_path)
	except Exception as err:
		print(str(err))
		print("**VGGFace model could not be loaded.")
		#ToDo Warning: vgg_face_weights.h5 file should be in weights folder
		
	vggfeda_descriptor = Model(inputs=vggfeda.layers[0].input , outputs=vggfeda.layers[-6].output)

	return vggfeda_descriptor


def loadOpenFace(): ## This is DeepFace actually
	print("Using OpenFace Adjusted to Ears (Input size 152x152)")
	
	model_path = os.path.join(ROOT_DIR, "weights", "openface_dsvggface2_600class_sgd1e5_156x156.h5")

	try:
		openface = load_model(model_path)
	except Exception as err:
		print(str(err))
		print("**OpenFace model could not be loaded.")
		#ToDo Warning: vgg_face_weights.h5 file should be in weights folder
		
	openface_descriptor = Model(inputs=openface.layers[0].input , outputs=openface.layers[-2].output)

	return openface_descriptor


## TwoStream Experiments 18/01/2021

def loadVGGFaceEar_TwoStream():
	print("Using VGGFaceEar TwoStream trained with 60 classes part of the VGGFace train set..." )
	
	model_path = os.path.join(ROOT_DIR, "weights", "twostream", "VGG_MS_60class_Avg_11.h5")

	try:
		vggfems = load_model(model_path)
	except Exception as err:
		print(str(err))
		print("**VGGFaceEar TwoStream model could not be loaded.")		
	
	## INPUT: inputs=[model_face.input, model_ear.input]
	vggfems_descriptor = Model(inputs=[vggfems.inputs[0], vggfems.inputs[1]] , outputs=vggfems.layers[-6].output)

	return vggfems_descriptor


def loadVGGFaceEar_TwoStream_2():
	print("Using VGGFaceEar TwoStream trained with 300 classes part of the VGGFace train set..." )
	
	# VGG_MS_300class_Concat_14.h5
	# VGG_MS_300class_Add_15.h5
	# VGG_MS_300class_Avg_16.h5

	model_path = os.path.join(ROOT_DIR, "weights", "twostream", "VGG_MS_300class_Concat_14.h5")

	try:
		vggfems = load_model(model_path)
	except Exception as err:
		print(str(err))
		print("**VGGFaceEar TwoStream model could not be loaded.")		
	
	## INPUT: inputs=[model_face.input, model_ear.input]
	vggfems_descriptor = Model(inputs=[vggfems.inputs[0], vggfems.inputs[1]] , outputs=vggfems.layers[-layer].output)

	return vggfems_descriptor


### SemiHardTripleLoss-based models

def loadVGGFaceEar_tripleloss():
	print("Using VGGFaceEar trained using triplet loss and 15 classes part of the VGGFace train set..." )
	
	model_path = os.path.join(ROOT_DIR, "weights", "tripleloss", "tripletloss_vggear.h5")

	try:
		vggfems = load_model(model_path)
	except Exception as err:
		print(str(err))
		print("**VGGFaceEar TwoStream model could not be loaded.")		
	
	## INPUT: inputs=[model_face.input, model_ear.input]
	vggfems_descriptor = Model(inputs=[vggfems.inputs[0]] , outputs=vggfems.layers[-1].output)

	return vggfems_descriptor



### UCSM Tests

def loadVGGUcsm():
	print("Using VGG16 trained with UCSM dataset ..." )
	
	model_path = os.path.join(ROOT_DIR, "weights", "ucsm", "vgg16_60c_ucsm_alllayers_opt3_3.h5")

	try:
		vggfems = load_model(model_path)
	except Exception as err:
		print(str(err))
		print("**VGG16 / short UCSM dataset.")		
	
	## INPUT: inputs=[model_face.input, model_ear.input]
	vggfems_descriptor = Model(inputs=[vggfems.inputs[0]] , outputs=vggfems.layers[-2].output)

	return vggfems_descriptor

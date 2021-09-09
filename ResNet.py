from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate, AveragePooling2D, BatchNormalization
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten, ZeroPadding2D, Add
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import optimizers
import tensorflow as tf

def res_identity(x, filters): 
	x_skip = x
	f1, f2 = filters
 
	x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer= ker_reg)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer= ker_reg)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=ker_reg)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Add()([x, x_skip])
	x = Activation('relu')(x)

	return x

def res_conv(x, s, filters):
	x_skip = x
  	f1, f2 = filters

  	x = Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=tf.keras.regularizers.L2(0.01))(x)
  	x = BatchNormalization()(x)
  	x = Activation('relu')(x)

  	x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.L2(0.01))(x)
  	x = BatchNormalization()(x)
  	x = Activation('relu')(x)

  	x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=tf.keras.regularizers.L2(0.01))(x)
  	x = BatchNormalization()(x)

  	x_skip = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=tf.keras.regularizers.L2(0.01))(x_skip)
  	x_skip = BatchNormalization()(x_skip)

  	x = Add()([x, x_skip])
  	x = Activation('relu')(x)

  	return x

def modelo_ResNet(tam_entrada, tam_salida,ker_mat= None):
	input_im = Input(shape=(300, 300, 3)) # cifar 10 images size
	x = ZeroPadding2D(padding=(3, 3))(input_im)


	x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = MaxPooling2D((3, 3), strides=(2, 2))(x)


	x = res_conv(x, s=1, filters=(64, 256))
	x = res_identity(x, filters=(64, 256))
	x = res_identity(x, filters=(64, 256))


	x = res_conv(x, s=2, filters=(128, 512))
	x = res_identity(x, filters=(128, 512))
	x = res_identity(x, filters=(128, 512))
	x = res_identity(x, filters=(128, 512))


	x = res_conv(x, s=2, filters=(256, 1024))
	x = res_identity(x, filters=(256, 1024))
	x = res_identity(x, filters=(256, 1024))
	x = res_identity(x, filters=(256, 1024))
	x = res_identity(x, filters=(256, 1024))
	x = res_identity(x, filters=(256, 1024))


	x = res_conv(x, s=2, filters=(512, 2048))
	x = res_identity(x, filters=(512, 2048))
	x = res_identity(x, filters=(512, 2048))

	x = AveragePooling2D((2, 2), padding='same')(x)

	x = Flatten()(x)

	if tam_salida== 1:
		x = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(x)
	else:
		x= Dense(tam_salida, activation='softmax', kernel_initializer='he_normal')(x)

	return Model(inputs=input_im, outputs=x, name='Resnet50')
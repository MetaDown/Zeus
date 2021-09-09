from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate, AveragePooling2D, BatchNormalization
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten, ZeroPadding2D
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
import tensorflow as tf

def bloque_1(entrada, filtros, kernel, ker_reg= None):
  x= Conv2D(filtros, kernel, padding= 'same', kernel_regularizer= ker_reg, activation = 'relu')(entrada)
  x= BatchNormalization()(x)
  x= Conv2D(filtros, kernel, padding= 'same', kernel_regularizer= ker_reg, activation = 'relu')(x)
  x= BatchNormalization()(x)
  x = MaxPooling2D(3, strides = 2, padding= "same")(x)

  return x

def bloque_2(entrada, filtros, kernel, ker_reg= None):
  x= Conv2D(filtros, kernel, padding= 'same', kernel_regularizer= ker_reg, activation = 'relu')(entrada)
  x= BatchNormalization()(x)
  x= Conv2D(filtros, kernel, padding= 'same', kernel_regularizer= ker_reg, activation = 'relu')(x)
  x= BatchNormalization()(x)
  x= Conv2D(filtros, kernel, padding= 'same', kernel_regularizer= ker_reg, activation = 'relu')(x)
  x= BatchNormalization()(x)
  x = MaxPooling2D(3, strides = 2, padding= "same")(x)

  return x


def bloque_3(entrada, neuronas, ker_reg= None):
  x= Dense(neuronas, kernel_regularizer= ker_reg)(entrada)
  x= Activation('relu')(x)
  return Dropout(0.5)(x)

def modelo_vgg16(tam_entrada, tam_salida,ker_mat= None):
	input_shape = Input(tam_entrada)

	b1= bloque_1(input_shape, 64, 3)
	b2= bloque_1(b1, 128, 3)
	b3= bloque_2(b2, 256, 3)
	b4= bloque_2(b3, 512, 3)
	b5= bloque_2(b4, 512, 3)
	si1 = Flatten()(b5)

	si2 = bloque_3(si1, 4096)
	si3 = bloque_3(si2, 1024)
	
	if tam_salida== 1:
		si4 = Dense(1)(si3, activation= 'sigmoid')
	else:
		si4 = Dense(tam_salida)(si3, activation= 'softmax')

	return Model(inputs= input_shape, outputs= si4, name= 'VGG16')
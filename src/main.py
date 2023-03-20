from tensorflow.keras.layers import Subtract,Input,Conv2D,BatchNormalization,Activation
from tensorflow.keras.models import Model
import tensorflow_datasets as tfds
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from data import extract_patches,join_patches,create_data
from model import DnCNN


'''
Performance metric of checking signal to noise ratio
'''
def PSNR(grnd_trth,image,max_val=1):
  ht,wd,channel=grnd_trth.shape
  wd=wd//40*40
  ht=ht//40*40
  mse=np.mean((grnd_trth-image)**2)
  if mse == 0:
    return 100
  return 20*np.log10(max_val/(np.sqrt(mse)))


'''
This function performs adding noise to the image given by Dataset
'''
def _parse_function(filename):
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)/255.

    noise_level=np.random.choice(NOISE_LEVELS)
    noisy_image=image+tf.random.normal(shape=(40,40,3),mean=0,stddev=noise_level/255)
    noisy_image=tf.clip_by_value(noisy_image, clip_value_min=0., clip_value_max=1.)

    return noisy_image,image

'''
This function performs the prediction of an image using our model
'''
def predict_fun(model,image_path,noise_level=30):
  #Creating patches for test image
  patches=extract_patches(image_path,40,[1])
  test_image=cv2.imread(image_path)

  patches=np.array(patches)
  ground_truth=join_patches(patches,test_image.shape)

  #predicting the output on the patches of test image
  patches = patches.astype('float32') /255.
  patches_noisy = patches+ tf.random.normal(shape=patches.shape,mean=0,stddev=noise_level/255) 
  patches_noisy = tf.clip_by_value(patches_noisy, clip_value_min=0., clip_value_max=1.)
  noisy_image=join_patches(patches_noisy,test_image.shape)

  denoised_patches=model.predict(patches_noisy)
  denoised_patches=tf.clip_by_value(denoised_patches, clip_value_min=0., clip_value_max=1.)

  #Creating entire denoised image from denoised patches
  denoised_image=join_patches(denoised_patches,test_image.shape)

  return patches_noisy,denoised_patches,ground_truth/255.,noisy_image,denoised_image


'''
This function plots some patches to show the difference 
between the noisy image and clear image
'''
def plot_patches(patches_noisy,denoised_patches):
  fig, axs = plt.subplots(2,10,figsize=(20,4))
  for i in range(10):

    axs[0,i].imshow(patches_noisy[i])
    axs[0,i].title.set_text(' Noisy')
    axs[0,i].get_xaxis().set_visible(False)
    axs[0,i].get_yaxis().set_visible(False)

    axs[1,i].imshow(denoised_patches[i])
    axs[1,i].title.set_text('Denoised')
    axs[1,i].get_xaxis().set_visible(False)
    axs[1,i].get_yaxis().set_visible(False)
  plt.show()


'''
This function compares sidewise the ground truth, noisy image and denoised image
'''
def plot_predictions(ground_truth,noisy_image,denoised_image):
  fig, axs = plt.subplots(1,3,figsize=(15,15))
  axs[0].imshow(ground_truth)
  axs[0].title.set_text('Ground Truth')
  axs[1].imshow(noisy_image)
  axs[1].title.set_text('Noisy Image')
  axs[2].imshow(denoised_image)
  axs[2].title.set_text('Denoised Image')
  plt.show()


def main():

  #Create data folder to store the patches
	os.mkdir('./data')
	os.mkdir('./data/patches')

	create_data('./images/train','./data/patches/train')
	print('Number of pactches obtained from train data : ',len(os.listdir('./data/patches/train')))

	create_data('./images/val','./data/patches/val')
	print('Number of pactches obtained from validation data : ',len(os.listdir('./data/patches/val')))

	dncnn= DnCNN()

	dncnn.compile(optimizer=tf.keras.optimizers.legacy.Adam(1e-03), loss=tf.keras.losses.MeanAbsoluteError())

	train_files=['data/patches/train/'+filename for filename in os.listdir('data/patches/train')]
	val_files=['data/patches/val/'+filename for filename in os.listdir('data/patches/val')]

	BATCH_SIZE=64
	NOISE_LEVELS=[15,25,20] 
	
	#Creating the Dataset
	train_dataset = tf.data.Dataset.from_tensor_slices(np.array(train_files)) 
	train_dataset = train_dataset.map(_parse_function)
	train_dataset = train_dataset.batch(BATCH_SIZE)

	val_dataset = tf.data.Dataset.from_tensor_slices(np.array(val_files))
	val_dataset = val_dataset.map(_parse_function)
	val_dataset = val_dataset.batch(BATCH_SIZE)

	dncnn.fit(train_dataset,shuffle=True,epochs=30,validation_data= val_dataset)

	patches_noisy,denoised_patches,ground_truth,noisy_image,denoised_image=predict_fun(dncnn,'images/val/102061.jpg',noise_level=25)
	print('PSNR of Noisy Image : ',PSNR(ground_truth,noisy_image))
	print('PSNR of Denoised Image : ',PSNR(ground_truth,denoised_image))
	plot_patches(patches_noisy,denoised_patches)

	plot_predictions(ground_truth,noisy_image,denoised_image)


if __name__=="__main__()":
	main

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

'''
This function extracts patches of a given size from the image
'''
def extract_patches(filepath, patch_sz,rescale_szs,save_dir=None):
  image=cv2.imread(filepath)

	#extracting imgname.jpg
  filename=filepath.split('/')[-1].split('.')[0] 
  
	#splitting the 3 pieces of information into their respective fields
  height,width,channels=image.shape 

	#we will be extracting the patches of the image
	#and store them
  patches=[]

	# for every size in rescale sizes, we will be rescaling the image
  for rescale_sz in rescale_szs:

    rescale_ht,rescale_wd=int(height*rescale_sz),int(width*rescale_sz)
		# we will be using cubic interpolation
    image_scaled=cv2.resize(image,(rescale_wd,rescale_ht), interpolation=cv2.INTER_CUBIC)
    
		for i in range(0,rescale_ht-patch_sz+1,patch_sz):
      for j in range(0,rescale_wd-patch_sz+1,patch_sz):
        x=image_scaled[i:i+patch_sz,j:j+patch_sz]

				# saving the patches in a directory
        if save_dir is not None:
          if not os.path.exists(save_dir):
            os.mkdir(save_dir)
          patch_filepath = save_dir+'/'+filename+'_'+str(rescale_ht)+'_'+str(i)+'_'+str(j)+'.jpg'
          cv2.imwrite(patch_filepath,x)

        patches.append(x)
  cv2_imshow(image)
  return patches


'''
This function joins the repsective patches previously formed to 
form a complete image
'''
def join_patches(patches,img_shp):
  image=np.zeros(img_shp)
  patch_sz=patches.shape[1]
  p=0
  for i in range(0,image.shape[0]-patch_sz+1,patch_sz):
    for j in range(0,image.shape[1]-patch_sz+1,patch_sz):
      image[i:i+patch_sz,j:j+patch_sz]=patches[p]
      p+=1
  return np.array(image)


'''
This function create data for all the images present in a directory
'''
def create_data(data_dir,save_dir=None):
  files_list=os.listdir(data_dir)
  print('Number of files in the '+data_dir+' is : '+str(len(files_list)))
  patch_size = 40
  crop_sizes = [1, 0.8, 0.7,0.5]
  data=[]
  for file in files_list:
    if file.endswith('.jpg'):
      patches=extract_patches(data_dir+'/'+file,patch_size,crop_sizes,save_dir)
      data+=patches
  return data


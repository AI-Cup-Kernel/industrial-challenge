import user_code.infer as infer
import matplotlib.pyplot as plt
import os
import numpy as np


N = 3


# get the list of images path 
image_names = []
for i in os.listdir('images'):
    image_names.append(os.path.join('images', i))

# read images as numpy array
imgs = [plt.imread(i) for i in image_names]

temp_pointer = 0
while temp_pointer < len(imgs):
    # get the next N images
    if temp_pointer + N > len(imgs):
        imgs_batch = imgs[temp_pointer:]
        #print(image_names[temp_pointer:])

    else:
        imgs_batch = imgs[temp_pointer:temp_pointer+N]
        #print(image_names[temp_pointer:temp_pointer+N])

    imgs_batch = np.array(imgs_batch)

    # load model
    model = infer.load_model()

    # infer
    result = infer.infer(model, imgs_batch)
    for i in range(result.shape[0]):
        temp = 'answer/'+image_names[i+temp_pointer].split('/')[-1][:-4] + '.txt'
        
        # make a text file with temp name in answers folder and write the result[i] in it
        
        # make temp file
        with open(temp, 'w') as f:
            for j in range(result.shape[1]):
                f.write(" ".join([str(t) for t in result[i][j]]) + '\n')
        
        #print(result[i].shape, temp)

    # print the result

    temp_pointer += N

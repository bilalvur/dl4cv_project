import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import rotate
import cv2
import gc
import torch.utils.data as data
import torch

'''Data normalization with substracting mean
   and dividing by standard deviation'''
def normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    normalized_data = (data - mean) / std
    return mean, std, normalized_data
 
'''Normalize with given mean and standard deviation'''
def normalizeWithValues(data, mean, std):
    return (data - mean) / std

'''Unflatten a vector'''
def unflattenBand(band, shape=(75, 75)):
    return np.array([np.array(b).astype(np.float32).reshape(shape) for b in band])

'''Create third channel with respect to
   given two channels and operation
   supported operations are 
       'average'
       'division'
       'substraction' '''
def createThirdChannel(band1, band2, operation='average'):
    new_channel = None
    if operation == 'average':
        new_channel = ((band1+band2)/2)[:, :]
    elif operation == 'division':
        new_channel = (band1/band2)[:, :]
    elif operation == 'substraction':
        new_channel = (band1-band2)[:, :]
    elif operation == 'multiplication':
        new_channel = (band1 * band2)
    else:
        #In case a different method is given
        new_channel = ((band1+band2)/2)[:, :]
    
    return new_channel
    
'''Rotate an image with angles from 0 up to 360'''
def createRotatedImages(image, angle):
    images = []
    for i in range(1, int(360/angle)):
        current_angle = i * angle
        rotated_image = rotate(input=image, angle=current_angle, reshape=False)
        images.append(rotated_image)
        
    return images
    
'''Create vertically and horizontally mirrored images'''
def createMirrorImages(image):
    mirror_images = []
    
    ud_image = np.flipud(image)
    lr_image = np.fliplr(image)
    
    mirror_images.append(ud_image)
    mirror_images.append(lr_image)
    
    return mirror_images

'''Create augmented images'''
def createAugmentedImages(image, rotation_angle, rotate=True, mirror=True):
    augmented_images = []
    
    all_images = []
    all_images.append(image)
    if rotate==True:
        rotated_images = createRotatedImages(image, rotation_angle)
        all_images.extend(rotated_images)
    
    mirror_images = []
    if mirror==True:
        for i in range(int(np.ceil(len(all_images)/2))):
            mirrored = createMirrorImages(all_images[i])
            mirror_images.extend(mirrored)
            
        all_images.extend(mirror_images)
    
    augmented_images.extend(all_images)
    
    return augmented_images

'''Read training data
   params:
    rotate - whether rotation should be done
    rotation_angle - should be able to divide 180 (e.g. 10, 30, 45, 90)
    mirror - whether mirroring should be done
    replaceNanWith -
        'mean': replace NaN inc_angle values with the mean of inc_angles
        'zero': replace NaN inc_angle values with zero'''
def getTrainData(rotate=True, rotation_angle=90, mirror=True, replaceNanWith='mean'):
    train = pd.read_json('train.json')

    #Read columns
    labels = train['is_iceberg']
    band1 = train['band_1']
    band2 = train['band_2']
    inc_angle = pd.to_numeric(train['inc_angle'], errors='coerce')
    ids = train['id']
    
    #Replace NaN values with given strategy
    if replaceNanWith=='mean':
        inc_angle = inc_angle.fillna(inc_angle.mean())
    elif replaceNanWith=='zero':
        inc_angle = inc_angle.fillna(0)
    elif replaceNanWith=='median':
        inc_angle = inc_angle.fillna(inc_angle.median())
    else:
        inc_angle = inc_angle.fillna(inc_angle.mean())
    
    number_of_images = len(labels)
    
    #Shape the bands in to (75, 75)
    #and create a third band
    unflattened_band1 = unflattenBand(band1)
    unflattened_band2 = unflattenBand(band2)
    unflattened_band3 = createThirdChannel(unflattened_band1, unflattened_band2, operation='average')
    
    #Normalize each band with their means and standard deviations
    mean1, std1, normalized_band1 = normalize(unflattened_band1)
    mean2, std2, normalized_band2 = normalize(unflattened_band2)
    mean3, std3, normalized_band3 = normalize(unflattened_band3)
    
    train_labels = []
    train_images = []
    train_inc_angles = []
    train_ids = []

    for i in range(number_of_images):
        #Do the augmentation
        augmented_images1 = createAugmentedImages(normalized_band1[i], rotate=rotate, rotation_angle=rotation_angle, mirror=mirror)
        augmented_images2 = createAugmentedImages(normalized_band2[i], rotate=rotate, rotation_angle=rotation_angle, mirror=mirror)
        augmented_images3 = createAugmentedImages(normalized_band3[i], rotate=rotate, rotation_angle=rotation_angle, mirror=mirror)
        
        number_of_augmented_images = len(augmented_images1)
        
        #Concatenate augmented images
        #Add labels and inc_angles repeteadly
        for j in range(number_of_augmented_images):
            concatenated_image = np.concatenate([augmented_images1[j][:,:,np.newaxis], augmented_images2[j][:,:,np.newaxis], augmented_images3[j][:,:,np.newaxis]], axis=-1)
            
            train_labels.append(labels[i])
            train_images.append(concatenated_image)
            train_inc_angles.append(inc_angle[i])
            train_ids.append(ids[i])
            
    #Transform lists to np arrays
    train_labels = np.asarray(train_labels)
    train_images = np.asarray(train_images)
    train_inc_angles = np.asarray(train_inc_angles)
    train_ids = np.asarray(train_ids)
    
    #Mean and standard deviation list
    means = [mean1, mean2, mean3]
    stds = [std1, std2, std3]

    return means, stds, train_images, train_labels, train_inc_angles, train_ids

def Augment(bandList,labels, rotate=True, rotation_angle=90, mirror=True):

    number_of_images = len(bandList[0])
    numberOfBands = len(bandList)
    train_labels = []
    augmentedBandList = []


    for i in range(number_of_images):
        # Do the augmentation

        augmented_images = []

        for j in range(numberOfBands):

            if (len(augmentedBandList)<=j ):
                augmentedBandList.append([])

            augmented_images = createAugmentedImages(bandList[j][i], rotate=rotate, rotation_angle=rotation_angle,
                                                  mirror=mirror)

            #augmented_images = np.array(augmented_images)

            augmentedBandList[j].extend(augmented_images)

            # Concatenate augmented images
            # Add labels and inc_angles repeteadly
        number_of_augmented_images = len(augmented_images)
        for k in range(number_of_augmented_images):
            train_labels.append(labels[i])


    # Transform lists to np arrays
    train_labels = np.asarray(train_labels)
    augmentedBandList = np.asarray(augmentedBandList)
    return (*augmentedBandList), train_labels

'''Read the test data
   Do the normalization with respect to training means and standard deviations'''
def getTestData(means, stds, replaceNanWith='mean'):
    test = pd.read_json('test.json')
    
    #Read columns
    band1 = test['band_1']
    band2 = test['band_2']
    inc_angle = pd.to_numeric(test['inc_angle'], errors='coerce')

    #Replace NaN values with given strategy
    if replaceNanWith=='mean':
        inc_angle = inc_angle.fillna(inc_angle.mean())
    elif replaceNanWith=='zero':
        inc_angle = inc_angle.fillna(0)
    elif replaceNanWith=='median':
        inc_angle = inc_angle.fillna(inc_angle.median())
    else:
        inc_angle = inc_angle.fillna(inc_angle.mean())

    #Shape the bands in to (75, 75)
    #and create a third band
    unflattened_band1 = unflattenBand(band1)
    unflattened_band2 = unflattenBand(band2)
    unflattened_band3 = createThirdChannel(unflattened_band1, unflattened_band2, operation='average')

    #Normalize each band with respective means and standard deviations from training data
    normalized_band1 = normalize(unflattened_band1, means[0], stds[0])
    normalized_band2 = normalize(unflattened_band2, means[1], stds[1])
    normalized_band3 = normalize(unflattened_band3, means[2], stds[2])

    
    concatenated_images = np.concatenate([normalized_band1[:,:,:,np.newaxis], normalized_band2[:,:,:,np.newaxis], normalized_band3[:,:,:,np.newaxis]], axis=-1)
    
    return concatenated_images, inc_angle

'''Visualize images with grid'''
def visualizeImages(images, fig_title, row=2, col=None, titles=None, cmap='gray'):
    if not col:
        col = int(np.ceil(len(images)/row))
    
    fig=plt.figure(num=fig_title)
    
    for i in range(len(images)):
        fig.add_subplot(row, col, i+1)
        plt.imshow(images[i], cmap=cmap)
        
        if titles != None:
            plt.title(titles[i])
    
    plt.show()

def smoothImages(images, kernel=(5,5), method='gauss', sigma=1):
    smoothed_images = []
    if method == 'gauss':
        for img in images:
            smoothed_images.append(cv2.GaussianBlur(img, ksize=kernel, sigmaX=sigma))
    elif method == 'average':
        for img in images:
            smoothed_images.append(cv2.blur(img, ksize=kernel))
    elif method == 'median':
        for img in images:
            smoothed_images.append(cv2.medianBlur(img, ksize=kernel[0]))
    elif method == 'bilateral':
        for img in images:
            smoothed_images.append(cv2.bilateralFilter(img, kernel[0], 75, 75))
    else:
        for img in images:
            smoothed_images.append(cv2.blur(img, ksize=kernel, sigma=sigma))
    
    return np.array(smoothed_images)
    
def smoothImage(img, kernel=(5,5), method='gauss', sigma=1):
    if method == 'gauss':
        return cv2.GaussianBlur(img, ksize=kernel, sigmaX=sigma)
    elif method == 'average':
        return cv2.blur(img, ksize=kernel)
    elif method == 'median':
        return cv2.medianBlur(img, ksize=kernel[0])
    elif method == 'bilateral':
        return cv2.bilateralFilter(img, 9, 75, 75)
    else:
        return cv2.blur(img, ksize=kernel, sigma=sigma)

# pixel normalization
def PixelNormalization(band):
    mean = np.mean(band, axis=0, keepdims=True)
    std = np.std(band, axis=0, keepdims=True)
    normalized_data = (band-mean)/std
    return mean, std, normalized_data

def GetTrainValidationDataForKFoldCrossValidation(train_index, val_index, bands, labels):

    numberOfBands = len(bands)

    trainData = [0]*numberOfBands
    valData = [0] * numberOfBands
    trainLabel = labels[train_index]
    valLabel = labels[val_index]

    for i in range(numberOfBands):
        trainData[i] = bands[i][train_index]
        valData[i] = bands[i][val_index]

        mean, std, trainData[i] = normalize(trainData[i])
        valData[i] = normalizeWithValues(valData[i], mean, std)

        gc.collect()

    *trainData, trainLabel = Augment(trainData, trainLabel)

    train_data = np.stack(trainData, axis=1)
    val_data = np.stack(valData, axis=1)

    gc.collect()

    return (IcebergData(train_data, trainLabel),
            IcebergData(val_data, valLabel))


class IcebergData(data.Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        img = self.X[index]
        label = self.y[index]

        img = torch.from_numpy(img)
        return img, label

    def __len__(self):
        return len(self.y)


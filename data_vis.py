import data_utils
import gc

if __name__ == '__main__':
    means, stds, train_images, train_labels, train_inc_angles, train_ids = data_utils.getTrainData(rotate=True,mirror=True)
    
    #Garbage collector
    gc.collect()
    
    #Visualize data augmentation
    titles = ['Original', 'Rotated-90', 'Rotated-180', 'Rotated-270', 'Original-V-Flipped', 'Original-H-Flipped', 'Rotated-90-V-Flipped', 'Rotated-90-H-Flipped']
    data_utils.visualizeImages(train_images[0:8,:,:,1], 'Augmentation', row=2, titles=titles, cmap='inferno')
    
    #Visualize different filters
    methods = ['gauss', 'average', 'median', 'bilateral']
    images = train_images[0,:,:,:]
    lis = []
    for i in range(3):
        lis.append(images[:,:,i])
        for j in range(len(methods)):
            lis.append(data_utils.smoothImage(images[:,:,i], kernel=(5,5), method=methods[j]))
    titles = ['Original', 'Gaussian', 'Average', 'Median', 'Bilateral','Original', 'Gaussian', 'Average', 'Median', 'Bilateral','Original', 'Gaussian', 'Average', 'Median', 'Bilateral']
    data_utils.visualizeImages(lis, 'Filters', row=3, titles=titles, cmap='inferno')
    
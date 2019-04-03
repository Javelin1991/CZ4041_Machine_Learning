# Importing some useful/necessary packages
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import cv2


def leaf_image(image_id,target_length=160):
    # `image_id` should be the index of the image in the images/ folder
    # Return the image of a given id(1~1584) with the target size (target_length x target_length)
    image_name = str(image_id) + '.jpg'
    leaf_img = plt.imread('images/'+image_name)  # Reading in the image
    leaf_img_width = leaf_img.shape[1]
    leaf_img_height = leaf_img.shape[0]
    #target_length = 160
    img_target = np.zeros((target_length, target_length), np.uint8)
    if leaf_img_width >= leaf_img_height:
        scale_img_width = target_length
        scale_img_height = int( (float(scale_img_width)/leaf_img_width)*leaf_img_height )
        img_scaled = cv2.resize(leaf_img, (scale_img_width, scale_img_height), interpolation = cv2.INTER_AREA)
        copy_location = int((target_length-scale_img_height)/2)
        img_target[copy_location:copy_location+scale_img_height,:] = img_scaled
    else:
        # leaf_img_width < leaf_img_height:
        scale_img_height = target_length
        scale_img_width = int( (float(scale_img_height)/leaf_img_height)*leaf_img_width )
        img_scaled = cv2.resize(leaf_img, (scale_img_width, scale_img_height), interpolation = cv2.INTER_AREA)
        copy_location = int((target_length-scale_img_width)/2)
        img_target[:, copy_location:copy_location+scale_img_width] = img_scaled

    return img_target


def species_image(species, labels, train_raw, classes):
    # `species` should be the index or species name
    # Return an image of a certain labeled species

    leaf_image_length = 160
    #img_target = np.zeros([leaf_image_length, 0], np.uint8)  # Initialization
    img_target = 240*np.ones([leaf_image_length, leaf_image_length*2], np.uint8)  # Initialization
    label_info = ''
    # if type(species)==int and species >= 0 and species < 99:
    # if species >= 0 and species < 99:
    if type(species)==int and species >= 0 and species < 99:
        images_index = np.where(labels==species)[0]
        label_info = str(species) + '-' + train_raw.species[images_index[0]]
    elif type(species)==str and species in classes:
        images_index = np.where(train_raw.species==species)[0]
        label_info = str(images_index[0]) + '-' + species
    else:
        print ('Error: Please input a valid index or species name')
        return


    for image_index in images_index:
        image_id = train_raw.id[image_index]
        leaf_img = leaf_image(image_id)
        img_target = np.append(img_target, leaf_img, axis=1)

    # Add information onto the first block
    cv2.putText(img_target, label_info, (10,90), cv2.FONT_HERSHEY_COMPLEX, 0.8, (100,170,0), 2)

    return img_target, label_info

def visualize_error(train_predictions, y_test):
    # Review the images where mistakes occur
    error_indices = np.where(train_predictions != y_test)[0];
    print ('The error indices: ', error_indices)
    for err_index in error_indices[0:]:
        print ('Error index in the test set: ', err_index)

        err_img_index = train_raw.id[test_index[err_index]]

        print ('Ground truth species index: {}'.format(y_test[err_index]))
        print ('Wrong predicting species index: {}'.format(train_predictions[err_index]))


        plt.imshow(leaf_image(err_img_index, 160), cmap='gray'); plt.axis('off'); plt.show()

        wrong_pred_species_img, label_info = species_image(train_predictions[err_index], labels, train_raw, classes)
        fig = plt.figure(num=None, figsize=(16, 3), dpi=1200, facecolor='w', edgecolor='w',frameon=False,linewidth = 0)
        wrong_pred_species_img = cv2.cvtColor(wrong_pred_species_img,cv2.COLOR_GRAY2RGB)
        wrong_pred_species_img = cv2.copyMakeBorder(wrong_pred_species_img,10,10,10,10,cv2.BORDER_CONSTANT,value=[255,0,0])

        plt.imshow(wrong_pred_species_img, cmap='gray'); plt.axis('off'); plt.show()

        ground_truth_species_img, label_info = species_image(y_test[err_index], labels, train_raw, classes)
        fig = plt.figure(num=None, figsize=(16, 3), dpi=1200, facecolor='w', edgecolor='w',frameon=False,linewidth = 0)
        plt.imshow(ground_truth_species_img, cmap='gray'); plt.axis('off'); plt.show()

        print ('#'*50)

def prepData(train, test, ss_split, labels):
    for train_index, test_index in ss_split.split(train, labels):
        X_train, X_test = train.values[train_index], train.values[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

    # Double check the data
    print (y_train.shape, y_test.shape)
    return X_train, X_test, y_train, y_test

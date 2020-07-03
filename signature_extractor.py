

import cv2
import matplotlib.pyplot as plt
from skimage import measure, morphology
from skimage.color import label2rgb
from skimage.measure import regionprops
import numpy as np
# read the input image


def extract(img):
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary

    # connected component analysis by scikit-learn framework
    blobs = img > img.mean()
    blobs_labels = measure.label(blobs, background=1)
    image_label_overlay = label2rgb(blobs_labels, image=img)

    fig, ax = plt.subplots(figsize=(10, 6))

    '''
    # plot the connected components (for debugging)
    ax.imshow(image_label_overlay)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    '''

    the_biggest_component = 0
    total_area = 0
    counter = 0
    average = 0.0
    for region in regionprops(blobs_labels):
        if (region.area > 10):
            total_area = total_area + region.area
            counter = counter + 1
        # print region.area # (for debugging)
        # take regions with large enough areas
        if (region.area >= 250):
            if (region.area > the_biggest_component):
                the_biggest_component = region.area

    average = (total_area/counter)
    print("the_biggest_component: " + str(the_biggest_component))
    print("average: " + str(average))

    # experimental-based ratio calculation, modify it for your cases
    # a4_constant is used as a threshold value to remove connected pixels
    # are smaller than a4_constant for A4 size scanned documents
    a4_constant = ((average/84.0)*250.0)+100
    print("a4_constant: " + str(a4_constant))

    # remove the connected pixels are smaller than a4_constant
    b = morphology.remove_small_objects(blobs_labels, a4_constant)
    # save the the pre-version which is the image is labelled with colors
    # as considering connected components
    plt.imsave('pre_version.png', b)

    # read the pre-version
    img = cv2.imread('pre_version.png', 0)
    # ensure binary
    img = cv2.threshold(img, 0, 255 , cv2.THRESH_OTSU)[1]
    H,W=img.shape

    #
    lines = cv2.HoughLines(img,1,np.pi/180,200)
    if(lines is not None):
        for line in lines:
            for rho,theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))

                img=cv2.line(img,(0,y1),(W,y2),(0,0,0),5)


    img = cv2.threshold(img, 0, 255 ,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    cv2.imwrite("./outputs/output.png", img)
    return img
def get_boxes(img):
    erode=cv2.erode(img,None,iterations=4)
    contours, h = cv2.findContours(erode, 1, 2)
    contour_sizes = [(cv2.contourArea(contour)) for contour in contours]
    print(contour_sizes)
    boxes=[]
    for i in range(len(contour_sizes)-1):
        if (contour_sizes[i]>=1000):
            x, y, w, h = cv2.boundingRect(contours[i])
            x2 = x + w
            y2 = y + h
            img=cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)
            boxes.append((x,y,x2,y2))
    cv2.imwrite("./outputs/output.png", img)
    return boxes

#
# img = cv2.imread('./inputs/test.png')[:,:,0]
# h,w = img.shape
# print(get_boxes(extract(img)))

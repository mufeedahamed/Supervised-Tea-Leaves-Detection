#paste everything in the C directory to run without changing any codes

import os
import string
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import mahotas as mt
import math

#Reading the dataset
dataset = pd.read_csv("good_tea_leaf_features.csv")
maindir = r'C:' #CHANGE THE DIRECTORY HERE
ds_path = maindir + "/tea_leaves"
img_files = os.listdir(ds_path)

#Creating Target Labels
breakpoints=[1,24,25,63]
target_list=[ ]
for file in img_files:
    target_num = int(file.split(".")[0])
    flag = 0
    i=0
    for i in range(0,len(breakpoints),1):
        if((target_num >= breakpoints[i]) and (target_num <= breakpoints[i+1])):
            flag = 1
            break
    if(flag==1):
        target = int((i/2))
        target_list.append(target)

y = np.array(target_list)
X = dataset.iloc[:,1:]

#Train test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 40)

#Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Applying SVM Classifier Model
clf = svm.SVC()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
metrics.accuracy_score(y_test, y_pred)

#Performing parameter tuning of the model
parameters = [{'kernel': ['rbf'], 'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5], 'C': [1, 10, 100, 1000]},
              {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
svm_clf = GridSearchCV(svm.SVC(decision_function_shape='ovr'), parameters, cv=5)
svm_clf.fit(X_train, y_train)
svm_clf.best_params_
means = svm_clf.cv_results_['mean_test_score']
stds = svm_clf.cv_results_['std_test_score']
#for mean, std, params in zip(means, stds, svm_clf.cv_results_['params']):
    #print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
y_pred_svm = svm_clf.predict(X_test)
#print(metrics.accuracy_score(y_test, y_pred_svm))

#Dimensionality Reduction using PCA
pca = PCA()
pca.fit(X)
var= pca.explained_variance_ratio_
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
plt.plot(var1)

#-----------------------------TESTING ARENA-----------------------------------------------------------------

test_img_path = 'C:/tea_leaves/45.png'
main_img=cv2.imread(test_img_path)
img=cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)

def feature_extract(img):
    names = ['area','perimeter','pysiological_length','pysiological_width','aspect_ratio','rectangularity','circularity', \
             'mean_r','mean_g','mean_b','stddev_r','stddev_g','stddev_b',
             'contrast','correlation','inverse_difference_moments','entropy'
            ]
    df = pd.DataFrame([], columns=names)
    gs = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gs, (15,15),0)
    ret_otsu,im_bw_otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((15,15),np.uint8)
    closing = cv2.morphologyEx(im_bw_otsu, cv2.MORPH_CLOSE, kernel)

#Shape features
    contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    M=cv2.moments(cnt)
    area=cv2.contourArea(cnt)
    perimeter=cv2.arcLength(cnt,True)
    x,y,w,h = cv2.boundingRect(cnt)
    aspect_ratio = float(w)/h
    rectangularity = w*h/area
    circularity = ((perimeter)*2)/area
    print(len(contours))

     #Color features
    red_channel = img[:,:,0]
    green_channel = img[:,:,1]
    blue_channel = img[:,:,2]
    blue_channel[blue_channel == 255] = 0
    green_channel[green_channel == 255] = 0
    red_channel[red_channel == 255] = 0

    red_mean = np.mean(red_channel)
    green_mean = np.mean(green_channel)
    blue_mean = np.mean(blue_channel)

    red_std = np.std(red_channel)
    green_std = np.std(green_channel)
    blue_std = np.std(blue_channel)

    #Texture features
    textures = mt.features.haralick(gs)
    ht_mean = textures.mean(axis=0)
    contrast = ht_mean[1]
    correlation = ht_mean[2]
    inverse_diff_moments = ht_mean[4]
    entropy = ht_mean[8]

    vector = [area,perimeter,w,h,aspect_ratio,rectangularity,circularity, red_mean,green_mean,blue_mean,red_std,green_std,blue_std,contrast,correlation,
                  inverse_diff_moments,entropy]
        
    df_temp = pd.DataFrame([vector],columns=names)
    df = df.append(df_temp)
    
    return df

features_of_img = feature_extract(img)
#print(features_of_img)

#--------------------------------prediction---------------------------------------
scaled_features = sc_X.transform(features_of_img)
#print(scaled_features)
#y_pred_mobile = svm_clf.predict(features_of_img)
y_pred_mobile = svm_clf.predict(scaled_features)
print(y_pred_mobile[0])


common_names = ['Pluck this', 'Avoid']

print(common_names[y_pred_mobile[0]])

#---------------------------------Labelling----------------------------------

disimg=main_img.copy()
gs = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
blur = cv2.GaussianBlur(gs, (15,15),0)
ret_otsu,im_bw_otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
kernel = np.ones((15,15),np.uint8)
closing = cv2.morphologyEx(im_bw_otsu, cv2.MORPH_CLOSE, kernel)
contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[0]
M = cv2.moments(cnt)
x,y,w,h = cv2.boundingRect(cnt)
cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

cxf=(x+w)/2
cx = math.trunc(cxf)
cyf=(y+h)/2
cy = math.trunc(cyf)

if y_pred_mobile[0]==0:
    cv2.rectangle(disimg,(x,y),(x+w,y+h),(255,0,0),4)
    cv2.putText(disimg, 'Pluck this', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 8)
elif y_pred_mobile[0]==1:
    cv2.rectangle(disimg,(x,y),(x+w,y+h),(0,0,255),4)
    cv2.putText(disimg, 'Avoid', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)



cv2.namedWindow('Test',cv2.WINDOW_NORMAL) 
cv2.imshow('Test', disimg)
cv2.waitKey()
cv2.destroyAllWindows()





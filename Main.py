import cv2
import numpy as np
from Classifying import *
from matplotlib import pyplot as plt
from featureExtraction import *
import cPickle
import argparse
import os

def show_sample_image(imageArr, index):
    image = np.squeeze(imageArr[index, :, :, :])
    cv2.imshow("sample", image)
    cv2.waitKey()


dataType = "PersonInvert"  # Right or Left or Person or PersonInvert

# Feature Selection...
ifGabor = True
ifPixel = True
ifHOG   = True
ifLBP   = True

if __name__ == "__main__":

    # Params ...

    parser = argparse.ArgumentParser('Passing different parameter for learning...')
    parser.add_argument("--dataPath", required=False, default='Data', help='path to data')
    parser.add_argument("--task", required=True, help='1:  diabete vs normal -  2: diabetic different types - 3:\
                                                      finding best window')
    parser.add_argument("--channel", required=True, help='h: H, s: S, v: V, r: R, g: G, b: B, G: gray')
    parser.add_argument("--downSampleSize", required=True, help='downsampling size for pixel feature')
    parser.add_argument("--isPlot", required=False, default='0', help='enable plotting. 1 for plotting and 0 for not-plotting')
    parser.add_argument("--saveIndex", required=False, default='0', help='saving new shuffled index for train-test splitting. 1 for saving and 0 for not saving')


    args = parser.parse_args()

    # Loading Datas...
    dataPath = args.dataPath
    if dataType == "Right":
        dataFolder = "R_split"
    elif dataType=="Left":
        dataFolder = "L_split"
    elif dataType=="Person":
        dataFolder = "personBase"
    elif dataType=="PersonInvert":
        dataFolder = "personBase_invert"

    controlImageArr = cPickle.load(open(os.path.join(dataPath, dataFolder, "controlImageArr.p"), "rb"))
    diabeteImageArr = cPickle.load(open(os.path.join(dataPath, dataFolder, "diabeteImageArr.p"), "rb"))
    drLabels = cPickle.load(open(os.path.join(dataPath, dataFolder, "DR_type.p"), "rb"))
    controlManner = cPickle.load(open(os.path.join(dataPath, dataFolder, "controllingManner.p"), "rb"))

    # show shapes of each array
    print diabeteImageArr.shape, controlImageArr.shape

    # show sample image from each data array
    if int(args.isPlot) == 1:
        show_sample_image(controlImageArr, 69)
        show_sample_image(diabeteImageArr, 69)

    # static region setting
    if args.task != '3':
        # regions = [[(395, 445), (35, 85), (275, 325), (635, 685)]]   # Cross of Andreas
        # regions = [[(65, 115), (381, 431), (235, 285), (605, 655)]]  # ourRegions
        # regions = [[(35, 85), (390, 440), (225, 275), (590, 640)],]  # ourRegions_2
        # regions = [[(75, 125), (375, 425), (235, 285), (615, 665)]]  # ourRegions_3
        # regions = [[(434, 464)]]                                     # Panceras
        regions = [[(75, 125), (375, 425), (235, 285), (615, 665)], [(395, 445), (35, 85), (275, 325), (635, 685)],
                   [(65, 115), (381, 431), (235, 285), (605, 655)]]  # comparison
        shrinkage = 0.51956
    else:
        rotationStep = 5
        windowLength = 50
        nPixel       = 720
        regions = []
        x_axis = []
        for i in range(0, nPixel-windowLength, rotationStep):
            regions.append([(i, i+windowLength)])
            x_axis.append(i)

        shrinkage = 0.51956


    # Normalizing & Constructing whole database
    nDiabetic = diabeteImageArr.shape[0]
    nControl = controlImageArr.shape[0]

    trainValPerc, testPerc = 1.0, 0.0
    nData = nDiabetic + nControl
    nTrain = np.uint16(np.ceil(trainValPerc * nData))


    meanAcc = [[], [], []] # for comparison

    for ii in range(1):
        if int(args.saveIndex) == 1:
            shuffleIdx = np.arange(nData)
            np.random.shuffle(shuffleIdx)
            np.savetxt("index.txt", shuffleIdx, fmt='%d')
        elif int(args.saveIndex) == 0:
            shuffleIdx = np.loadtxt("index.txt", dtype=np.uint16)

        errLogs = []

        for regIdx, reg in enumerate(regions):
            diabeticLabel = []
            isDiabeteLabel = []
            irisData = []
            reg_mean_acc = 0

            ## Adding diabetic images
            for i in range(nDiabetic):
                img = np.squeeze(diabeteImageArr[i, :, :, :])
                if args.channel in ['h', 's', 'v']:
                    hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    rawImage = hsvImage[:, :, ['h', 's', 'v'].index(args.channel)]
                elif args.channel in ['b', 'g', 'r']:
                    rawImage = img[:, :, ['b', 'g', 'r'].index(args.channel)]
                else:
                    rawImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                rawImage = np.squeeze(rawImage)
                img_normalized = np.copy(rawImage)
                img_normalized = cv2.equalizeHist(img_normalized)

                imageFeature = []
                if ifGabor:
                    imageFeature = imageFeature + gaborFeature(img_normalized, reg)
                if ifHOG:
                    imageFeature = imageFeature + hogFeature(img_normalized, reg)
                if ifLBP:
                    imageFeature = imageFeature + lbpFeature(img_normalized, reg)
                if ifPixel:
                    imageFeature = imageFeature + extract_image_feature(img_normalized, reg, np.float16(args.downSampleSize))


                imageFeature_arr = np.squeeze(np.array(imageFeature, dtype=np.float32))
                irisData.append(reshape(imageFeature_arr, (-1, )))
                isDiabeteLabel.append(1)
                diabeticLabel.append(drLabels[i])

            ## Adding control images
            for i in range(nControl):
                img = np.squeeze(controlImageArr[i, :, :, :])
                if args.channel in ['h', 's', 'v']:
                    hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    rawImage = hsvImage[:, :, ['h', 's', 'v'].index(args.channel)]
                elif args.channel in ['b', 'g', 'r']:
                    rawImage = img[:, :, ['b', 'g', 'r'].index(args.channel)]
                else:
                    rawImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                rawImage = np.squeeze(rawImage)
                img_normalized = np.copy(rawImage)
                img_normalized = cv2.equalizeHist(img_normalized)  # concat all regions and normalize...(new idea)


                imageFeature = []
                if ifGabor:
                    imageFeature = imageFeature + gaborFeature(img_normalized, reg)
                if ifHOG:
                    imageFeature = imageFeature + hogFeature(img_normalized, reg)
                if ifLBP:
                    imageFeature = imageFeature + lbpFeature(img_normalized, reg)
                if ifPixel:
                    imageFeature = imageFeature + extract_image_feature(img_normalized, reg,
                                                                        np.float16(args.downSampleSize))


                imageFeature_arr = np.squeeze(np.array(imageFeature, dtype = np.float32))
                irisData.append(reshape(imageFeature_arr, (-1, )))
                isDiabeteLabel.append(0)

            irisData = np.array(irisData, dtype=np.float32)
            isDiabeteLabel = np.array(isDiabeteLabel, dtype=np.float32)
            diabeticLabel = np.array(diabeticLabel, dtype=np.float32)
            print "All Data: ", irisData.shape

            # train-test-val splitting
            X_train = irisData[shuffleIdx[:nTrain], :]
            Y_train = isDiabeteLabel[shuffleIdx[:nTrain]]
            X_test = irisData[shuffleIdx[nTrain:], :]
            Y_test = isDiabeteLabel[shuffleIdx[nTrain:]]

            np.savetxt("sampleFeature.txt", X_train)

            ## Show dimension of train and test ...
            print "Train Data:", X_train.shape, Y_train.shape
            print "Test  Data:", X_test.shape, Y_test.shape

            ## LDA Classifing
            tuned_parameters = [{'solver': ['lsqr'], 'shrinkage':np.linspace(0.05, 0.95, 70)}]
            if args.task == '1':
                # acc_mlp, recall_mlp, percision_mlp = mlp_classify(X_train, Y_train, X_test, Y_test)   # MLP
                # acc_lsvm, recall_lsvm, percision_lsvm = linearsvm_classify(X_train, Y_train, X_test, Y_test) # Linear-SVM
                # acc_rf, recall_rf, percision_rf = rf_classify(X_train, Y_train, X_test, Y_test) # Random Forrest
                acc_ada, recall_ada, percision_ada = adaboost_classify(X_train, Y_train, X_test, Y_test) # Adaboost
                meanAcc[regIdx].append(acc_ada)
                # acc_ens, recall_ens, percision_ens = ensemble_classify_hardVoting(X_train, Y_train, X_test, Y_test) # Ensemble Classifier
                # acc_lda, recall_lda, percision_lda = lda_classify(X_train, Y_train, tuned_parameters,\
                #               score='accuracy', k_fold_num=5, shrinkage=shrinkage) # LDA


            elif args.task == '3':
                test_train_ratio = 0.2
                x_train_new = irisData[shuffleIdx[:np.uint32(np.floor(nTrain*test_train_ratio))], :]
                y_train_new = isDiabeteLabel[shuffleIdx[:np.uint32(np.floor(nTrain*test_train_ratio))]]
                x_val = irisData[shuffleIdx[np.uint32(np.floor(nTrain*test_train_ratio)+1):], :]
                y_val = isDiabeteLabel[shuffleIdx[np.uint32(np.floor(nTrain*test_train_ratio))+1:]]

                # y_true, y_pred = lda_classify2(x_train_new, y_train_new,x_val, y_val, tuned_parameters, score='accuracy',
                #                         k_fold_num=5, shrinkage=shrinkage)
                bdt = AdaBoostClassifier(algorithm="SAMME", n_estimators=30)
                bdt.fit(x_train_new, y_train_new)
                y_true, y_pred = y_val, bdt.predict(x_val)

            if args.task == '3':
                errLogs.append(np.mean((y_true != y_pred)))



        if args.task == '3':
            p30 = np.poly1d(np.polyfit(np.array(x_axis), np.array(errLogs), deg=15))

            if int(args.isPlot) == 1:
                plt.figure(ii)
                plt.plot(x_axis, p30(np.array(x_axis)))
                plt.plot(x_axis, np.array(errLogs))
                plt.show()

    print mean(meanAcc[0]), mean(meanAcc[1]), mean(meanAcc[2])

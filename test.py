"""
refer to https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py
"""
import numpy
import numpy as np
import os
import cv2

__all__ = ['SegmentationMetric']

"""
confusionMetric  # 注意：此处横着代表预测值，竖着代表真实值，与之前介绍的相反
P\L     P    N
P      TP    FP
N      FN    TN
"""


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(np.transpose(self.confusionMatrix)) / np.transpose(self.confusionMatrix).sum(axis=1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def meanIntersectionOverUnion(self):
        # Intersection = TP    Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        mIoU = np.nanmean(IoU)  # 求各类别IoU的平均
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):  # 同FCN中score.py的fast_hist()函数
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        # num_class * gt + pred
        # [ 0  4 10  0  5 11 10  5 15]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))


if __name__ == '__main__':



        metric = SegmentationMetric(2)  # 类别数


        # imgPredict = np.array([[0,1,2,2],
        #                        [0,1,2,2],
        #                        [0,1,2,2],
        #                        [0,1,2,2]])  # 可直接换成预测图片
        #
        # imgLabel = np.array([[0,1,2,2],
        #                      [1,2,0,0],
        #                      [0,1,2,2],
        #                      [0,1,2,2]])  # 可直接换成标注图片

        imgPredict = cv2.imread("./1.png")
        imgPredict = np.transpose(imgPredict, [2, 0, 1])  # uint8 (3, 340, 340)
        imgPredict = imgPredict[:][:][0]
        # print(pred_data.dtype)

        imgLabel = cv2.imread("./2.png")
        imgLabel = np.transpose(imgLabel, [2, 0, 1])  # uint8 (3, 340, 340)
        imgLabel = imgLabel[:][:][0]

        imgPredict = np.where(imgPredict > 2, 1, 0)
        imgLabel = np.where(imgLabel > 2, 1, 0)


        metric.addBatch(imgPredict, imgLabel)

        print('ConfusionMatrix :')
        print(metric.confusionMatrix)  # numpy.transpose() 矩阵转置

        print('Add:')
        print(numpy.sum(metric.confusionMatrix, axis=0))

        print('%:')
        print(metric.confusionMatrix / numpy.sum(metric.confusionMatrix, axis=0))
        # print('ConfusionMatrix :')
        # print(numpy.transpose(metric.confusionMatrix))

        pa = metric.pixelAccuracy()
        cpa = metric.classPixelAccuracy()
        mpa = metric.meanPixelAccuracy()
        mIoU = metric.meanIntersectionOverUnion()
        print('pa is : %f' % pa)

        print('cpa is :')  # 列表
        print(cpa)

        print('mpa is : %f' % mpa)
        print('mIoU is : %f' % mIoU)








#           GT
#       0  1  2  3
#    0[[2. 1. 0. 0.]
#    1 [0. 2. 0. 0.]
# P  2 [0. 0. 2. 0.]
#    3 [0. 0. 1. 1.]]
# pa is : 0.777778
# cpa is :
# [1.         0.66666667 0.66666667 1.        ]
# mpa is : 0.833333
# mIoU is : 0.625000
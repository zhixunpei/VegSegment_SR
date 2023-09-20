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
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
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

    with open('./all_vallist.txt', 'r') as li:
        data_list = [line.replace('\n', '.png') for line in li]
    data_list.sort()
    print(f'The number of val_data:{len(data_list)}')

    model = ['MST++', 'MPRnet', 'HSCNN++', 'HRnet', 'HDnet']
    file_path = './res/vi low -0.2/all/'
    epoch = 20

    for mod in model:
        print(f'----------now: {mod}----------')
        metric = SegmentationMetric(2)  # 类别数

        for i in range(len(data_list)):
            pred_data = cv2.imread(os.path.join(file_path, f'{mod}/{epoch}', data_list[i]))
            pred_data = np.transpose(pred_data, [2, 0, 1])  # uint8 (3, 340, 340)
            pred_data = pred_data[:][:][0]
            # print(pred_data.dtype)

            gt_data = cv2.imread(os.path.join(f'./miou/all/Val_GT', data_list[i]))
            gt_data = np.transpose(gt_data, [2, 0, 1])  # uint8 (3, 340, 340)
            gt_data = gt_data[:][:][0]
            # print(rgb_data.dtype)

            # print(f'{data_list[i]}')
            # imgPredict = np.array([[0,0,2],
            #                         [0,1,3],
            #                         [2,1,3]])  # 可直接换成预测图片
            # imgLabel = np.array([[0,1,2],
            #                       [0,1,2],
            #                       [2,1,3]])  # 可直接换成标注图片
            imgPredict = np.where(pred_data > 2, 1, 0)
            imgLabel = np.where(gt_data > 2, 1, 0)
            metric.addBatch(imgPredict, imgLabel)

        print('ConfusionMatrix :')
        print(numpy.transpose(metric.confusionMatrix))  # numpy.transpose() 矩阵转置

        print('Add:')
        print(numpy.sum(numpy.transpose(metric.confusionMatrix), axis=0))

        print('%:')
        print(numpy.transpose(metric.confusionMatrix) / numpy.sum(numpy.transpose(metric.confusionMatrix), axis=0))
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

        with open(os.path.join(file_path, f'{mod}/{epoch}', "res.txt"), mode='w', encoding='utf-8') as f:
            f.write(os.path.join(file_path, f'{mod}/{epoch}'))
            f.write('\n')
            f.write('ConfusionMatrix :\n')
            f.write(str(numpy.transpose(metric.confusionMatrix)))
            f.write('\n')
            f.write('%:\n')
            f.write(str(numpy.transpose(metric.confusionMatrix) / numpy.sum(numpy.transpose(metric.confusionMatrix), axis=0)))
            f.write('\n')
            f.write('pa is : %f\n' % pa)
            f.write('cpa is :\n')  # 列表
            f.write(str(cpa))
            f.write('\n')
            f.write('mpa is : %f\n' % mpa)
            f.write('mIoU is : %f\n' % mIoU)






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
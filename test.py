# _*_ coding: utf-8 _*_
# @File : test.py
# @Time : 2021/10/30 9:46
# @Author : Yan Qiuxia
# import numpy as np
# arr1 = np.array([1,1,0]).reshape(3,1)
# print(arr1.shape)
# arr2 = np.array([[1,3],[2,4],[4,9]])
# print(arr2)
# print(arr2.shape)
# c = arr2*arr1
# print(c)
import numpy as np
# span_pred_id = [0,1,1]
# pred_idxs = np.where(span_pred_id == 1)[0]
# print(pred_idxs)
# for pred_idx in pred_idxs:
#     print(type(pred_idx))
#     print(pred_idx)


# np.random.seed(2021)
# print(np.random.random())
# print(np.random.random())
#
import torch
# torch.manual_seed(2021)
# print(torch.rand(2))
# print(torch.rand(2))
# for i in range(10):
#     sch_p = 10 / (10 + np.exp(i))
#     print(sch_p)
#
# spans_dranges = torch.rand((4,3,2))
#
# spans_starts = spans_dranges[:,:,0]
# print(spans_starts.shape)
#
# spans_ends = spans_dranges[:,:,1]
# print(spans_ends.shape)

a = [1,2,3,4]
b = [0,2]
a+=b
print(a)

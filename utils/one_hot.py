import torch
import numpy as np

# gt = np.random.randint(0,5, size=[2,1,15,15])  #先生成一个15*15的label，值在5以内，意思是5类分割任务
# gt = torch.LongTensor(gt)



# def get_one_hot(label, N):
#     size = list(label.size())
#     label = label.view(-1)   # reshape 为向量
#     ones = torch.sparse.torch.eye(N)
#     ones = ones.index_select(0, label)   # 用上面的办法转为换one hot
#     size.append(N)  # 把类别输目添到size的尾后，准备reshape回原来的尺寸
#     return ones.view(*size)

# gt_one_hot = get_one_hot(gt, 5)

# print(gt_one_hot)
# print(gt_one_hot.shape)

# print(gt_one_hot.argmax(-1) == gt)  # 判断one hot 转换方式是否正确，全是1就是正确的

# 一：当维度为 N  1 *
# one-hot后 N C *
 
def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, torch.LongTensor(input), 1)
 
    return result

# gt = np.random.randint(0,2, size=[2,1,15,15])  #先生成一个15*15的label，值在5以内，意思是5类分割任务
# gt = torch.LongTensor(gt)

# gt_one_hot = make_one_hot(gt, 2)

# print(gt_one_hot)
# print(gt_one_hot.shape)

# print(gt_one_hot[0,0].argmax(-1) == gt[0,0])  # 判断one hot 转换方式是否正确，全是1就是正确的


# 二：当维度为 1 * 
# one_hot后 N *
# def make_one_hot(input, num_classes):
#     """Convert class index tensor to one hot encoding tensor.
#     Args:
#          input: A tensor of shape [N, 1, *]
#          num_classes: An int of number of class
#     Returns:
#         A tensor of shape [N, num_classes, *]
#     """
#     shape = np.array(input.shape)
#     shape[0] = num_classes
#     shape = tuple(shape)
#     result = torch.zeros(shape)
#     result = result.scatter_(0, torch.LongTensor(input), 1)
 
#     return result
 
# * 代表图像大小 例如 224 x 224
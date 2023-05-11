import torch
import torch.nn.functional as F


# def gen(image):
#     image = F.interpolate(image, scale_factor=0.125)
#     # image[image > 1] = 1
#     # image[image < 0] = 0
#     i_front = image.reshape((image.shape[0], image.shape[1], -1))
#     i_back = 1 - i_front
#     fuse = torch.cat((i_front, i_back), dim=1)
#     # fuse_t = torch.zeros([fuse.shape[0], fuse.shape[2], fuse.shape[1]]).cuda()
#     # for i in range(fuse.shape[0]):
#     #     fuse_t[i] = fuse[i].t()
#     fuse_t=fuse.transpose(2,1)
#     res = torch.bmm(fuse_t, fuse)
#     return res


def make_one_hot(input, x,num_classes=2,flag=0):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    #shape = np.array(input.shape)
    if flag==0:

        input = F.interpolate(input, scale_factor=x,mode="bilinear")
    else:
        pass
    # shape = input.shape
    # shape[1] = num_classes
    # shape = tuple(shape)
    input=input.to(torch.int64)
    result=torch.nn.functional.one_hot(input, num_classes=2)
    # result = torch.zeros((input.shape[0],num_classes,input.shape[2],input.shape[3])).cuda()
    # result = result.scatter_(1, torch.LongTensor(input), 1)

    #result=torch.squeeze(result)
    result=result.squeeze(dim=1)
    result1=result.reshape((result.shape[0], result.shape[3], -1))# transfer N C H*W
    result2=result1.permute(0,2,1)
    result1=result1.to(torch.float32)
    result2=result2.to(torch.float32)

    A = torch.bmm(result2, result1) 
    A= torch.unsqueeze(A,1)
    return A


# if __name__ == "__main__":
#     img = torch.rand([6, 1, 288, 288]).cuda()
#     output = gen(img)
#     print(output)

# coding:utf-8
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import time

THRESHOLD = 200  # for white background
TABLE = [1]*THRESHOLD + [0]*(256-THRESHOLD)

def line_split(image, table=TABLE, split_threshold=4):

    if not isinstance(image, Image.Image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        else:
            raise TypeError

    image_ = image.convert('L')
    bn = image_.point(table, '1')
    #bn.save("0_二值化后图像.jpg")
    bn_mat = np.array(bn)
    h, pic_len = bn_mat.shape   #(1944, 1325)

    # 修改思路：不用全0判断，而是用连续性判断，文字行必然是Ture,False交替出现
    # 因此它的diff值一定经常有1，因为True-Fasle = False-True = True = 1 而全黑全白行diff之后应该全是0
    bn_mat_diff = np.diff(bn_mat)
    project = np.sum(bn_mat_diff, 1) #(1944,)
    pos = np.where(project <= split_threshold)[0]  # split_threshold不能在设置为0了,10左右应该可以，可以自己调
    project[pos] = 0
    # 正反傅里叶，将信号连续化
    transformed=np.fft.fft(project) #傅里叶变换
    itransformed_real = np.real(np.fft.ifft(transformed))
    signal = np.around(itransformed_real)
    #plt.plot(signal)
    #plt.savefig("1_滤波后信号.jpg")
    pos = np.where(signal <= 0)[0]

    # diff函数就是执行的是后一个元素减去前一个元素
    diff = np.diff(pos)
    #print(diff)

    coordinate = list(zip(pos[:-1], pos[1:]))
    info = list(zip(diff, coordinate))
    info = list(filter(lambda x: x[0] > 10, info))
    line_res = []
    for pos_info in info:
        width = pos_info[0]
        x1, y1, x2, y2 = 0, pos_info[1][0]-int(0.1*width), pic_len, pos_info[1][1]+int(0.1*width)
        sub = image.crop((x1, y1, x2, y2))
        line_res.append([np.array(sub), (x1, y1, x2, y2)])
    return line_res

###################################test#########################

img = cv2.imread('test5.jpg')
t1 = time.time()
line_imgs = line_split(img)
t2 = time.time()
print("Cost time: {}".format(t2-t1))
tmp = 0
for line_img in line_imgs:
    cv2.imwrite('./debug/'+str(tmp)+'.jpg',line_img[0])
    tmp += 1
    start = line_img[1][0],line_img[1][3]
    end = line_img[1][2],line_img[1][3]
    cv2.line(img,start,end,(0,0,255),1)
cv2.imwrite('line_split_result.jpg',img)

    


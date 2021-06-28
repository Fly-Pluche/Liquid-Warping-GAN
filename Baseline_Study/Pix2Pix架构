## Pix2Pix架构

![image-20210615181501766](https://gitee.com/Black_Friday/blog/raw/master/image/image-20210615181501766.png)

### Auto-encoder

![image-20210615181636376](C:\Users\BlackFriday\AppData\Roaming\Typora\typora-user-images\image-20210615181636376.png)

上图的code是Encoder提取的图像信息。

我们经常会遇到这些编码-解码的网络结构，其中的Encoder部分就是对图像进行一个底层特征的提取，然后得到code.

这里的code是`一维的`,其表达的含义可以理解为经过全连接后输出的向量：

第一个数据是这个图片是否有头，

第二个数据是这个图片是否有尾巴

and so on

而Decoder就是一个还原的逆过程。

### 改进

而为了让输入的图片与输出的图片越接近越好，Pix2Pix该进了U-Net，即将Encoder与Decoder通过残差相连…..

想想$U_2-net$也是一样骚操作。。。

![image-20210615182113052](https://gitee.com/Black_Friday/blog/raw/master/image/image-20210615182113052.png)


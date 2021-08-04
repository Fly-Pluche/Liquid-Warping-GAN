# Free-Form Image Inpainting with Gated Convolution

> >  GitHub:https://github.com/JiahuiYu/generative_inpainting

>
>
>## Abstract
>
>We present a generative image inpainting system to complete images with free-form mask and guidance. The system is based on gated convolutions learned from millions of images without additional labelling efforts. The proposed gated convolution solves the issue of vanilla convolution that treats all input pixels as valid ones, generalizes partial convolution by providing a learnable dynamic feature selection mechanism for each channel at each spatial location across all layers. Moreover , as free-form masks may appear anywhere in images with any shape, global and local GANs designed for a single rectangular mask are not applicable. Thus, we also present a patch-based GAN loss, named SNPatchGAN, by applying spectral-normalized discriminator on dense image patches. SN-PatchGAN is simple in formulation, fast and stable in training. Results on automatic image inpainting and user-guided extension demonstrate that our system generates higher-quality and more flexible results than previous methods. Our system helps user quickly remove distracting objects, modify image layouts, clear watermarks and edit faces.
>
>> 提出了一个图像修复网络。由于污染的图像，并且采用门控卷积解决了普通卷积（将所有输入的像素都视为有效像素，因此不适用与图像的空洞填充）的问题，通在所有层的每个空间位置的每个通道提供可学习的动态特征，推广部分卷积。
>>
>> 由于会在任意位置出现不同形状的masks，为单一的矩形掩码设计一个全局和局部的GANs不会奏效。故此，设计了一个patch-base的GAN损失，叫SNPatchGAN，通过在密集图像斑块上使用光谱归一化鉴别器。



## USING

可删除分心的对象，修改图像布局，去除水印，编辑脸部。

![image-20210803162226062](https://gitee.com/Black_Friday/blog/raw/master/image/image-20210803162226062.png)![image-20210803174252917](https://gitee.com/Black_Friday/blog/raw/master/image/image-20210803174252917.png)

## Intrduction

在计算机视觉中存在两种图像补全的方法

- 依靠低层次的特征图匹配
- 使用深度卷积网络的前馈生成模型

前一种方法可以用于合成简单的纹理，但是不适用于复杂的环境。

而后一种方法则通过学习语义，依靠端到端的方式合成图像，较为优秀。

# Approach

## 3.1 Gated Convolution

### 普通卷积

先来解释一下为什么普通卷积不适用于图像填补。

我们先考虑一个卷积层，其中一组滤波器作为输出应用于输入特征映射（We consider a convolutional layer in which a bank of filters are applied to the input feature map as output.）。

假设输入为通道为C，则每一个在C通道中位于（y,x）的像素可以被这样计算。
$$
O_{y,x}=\sum^{k'_h}_{i=-k'_h}\sum^{k'_w}_{i=-k'_w}W_{k'_h+i,k'_w+j}·I_{y+i,x+j}
$$

- $I_{y+i,x+j}∈R^C$​是输入
- $W∈R^{k_h×k_w×C'×C}$代表卷积过滤器
- $O_{y,x}$是输出
- 其中的$x,y$​​代表输出map的x轴y轴
- $k_h,k_w$​是卷积核的尺寸（e.g.3×3）,$k'_h=\frac{k_h-1}{2}，k'_w=\frac{k_w-1}{2}$

该方程表明，对于所用的空间位置（x,y）相同的滤波器被用于产生普通卷积层的输出。

这让图像分类和目标检测这样的任务是有意义的，其中输入的所有像素是有效的。然后图像修补中，输入的图像是由有效像素与无效像素组合而成。这回导致生成伪影，颜色差异，模糊和明显的边缘反应等。

### 部分卷积

故此，提出了`部分卷积`，采用了掩码和重归一化去让卷积只依赖于有效的像素
$$
O_{x,y}=\left\{
\begin{array}{**lr**}
\sum \sum W（I⊙\frac{M}{sum(M)}）,if&sum(M)>0
\\0,&otherwise
\end{array}
\right.
$$

- 其中M是对应的二进制掩码，1代表位置（y,x）有效，0表示无效

- ⊙表示逐元素乘法

  在每个部分卷积操作之后，掩码更新步骤需要按照如下规则传播:
  $$
  m'_{y,x}=1,如果sum(M)>0
  $$
  部分卷积虽然提高了不规则掩着色质量，但是仍然存在问题：

  - 启发式的将所有的空间位置分类为有效或者无效，不管有多少像素被前一层的过滤范围覆盖，下一层的mask将被设置为1（例如，1个有效像素与9个有效像素被视为更新当前mask的相同对象）
  - 与额外的用户输入不兼容。他们的目的是映入一个图像绘制系统，用户可以选择在其中绘制草图作为条件通道，但是这些像素的位置不知道算有效还是无效，并且不知道如何正确的更新到下一层的蒙版
  - 无效像素在深层会逐渐消失，逐渐将所有掩码值转换为1，然而，我们的研究表明，如果我们允许网络自动学习最佳掩码，即使在深层，网络也会为每个空间位置分配软掩码值。
  - 每一层的所有通道共享同一个掩码，限制了灵活性。

![image-20210803214703411](C:\Users\BlackFriday\AppData\Roaming\Typora\typora-user-images\image-20210803214703411.png)

### 门控卷积

从本质上讲，部分卷积可以看作是不可学习的单通道特征硬门控。

而他们后续提出的门控卷积是可以通过学习改变的软门控卷积。
$$
Gating_{y,x}=\sum{}\sum{}W_g·I\\
Feature_{y,x}=\sum{}\sum W_f·I\\
O_{y,x}=φ(Feature_{y,x})⊙σ(Gating_{y,x})
$$
其中σ是sigmoid函数，φ可以是任意的激活函数（例如ReLU，LeakyReLU，ELU扥扥）

$W_g和W_f$是两个不同的卷积核

f是mask，g是input




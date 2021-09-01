# 2020 CVPR || Recurrent Feature Reasoning for Image Inpainting

> 不同的图像修补网络有不同的特效，这个网络的对象是主体大部分缺失的图像。
>
> > [Code](https://github.com/jingyuanli001/RFR-Inpainting)and [Paper]([Recurrent Feature Reasoning for Image Inpainting (thecvf.com)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Recurrent_Feature_Reasoning_for_Image_Inpainting_CVPR_2020_paper.pdf))

## Abstract

现有的图像修补方法在修复规则或小的图像缺陷方面效果较好，由于大孔洞中心缺少相应的信息，网络在大范围缺失的图像填充中，表现较差。

对此，提出了设计了一个Recurrent Feature Reasoning（RFR）网络，其主要由即插即用的特征推理模块和Knowledge Consistent Attention（KCA）模块构成。

![image-20210808121005384](C:\Users\BlackFriday\AppData\Roaming\Typora\typora-user-images\image-20210808121005384.png)

注意上图中被修复图片的空白部分。

### 思想

递归的思想与我们解决问题的方式类似，由易到难。由简单的部分的结果作为推理困难部分的依据。

孔洞边界$\longrightarrow$简单问题

孔洞中心$\longrightarrow$困难问题

为了捕捉来自遥远地方的信息，提出了KCA注意力并将其纳入RFR。



## Introduction

对于先阶段的网络，如果采用递归循环的方式修补网络。要么是性能不佳（迭代都会将feature map映射回RGB空间，导致信息失真(如将128×128×64 feature map转换为256×256×3 RGB  image)。），要么是计算量偏大。

<img src="C:\Users\BlackFriday\AppData\Roaming\Typora\typora-user-images\image-20210808120044284.png" alt="image-20210808120044284" style="zoom:50%;" />

与现有的渐进方法不同，RFR模块在特征映射空间中执行这种渐进过程，不仅保证了优越的性能，而且解决了网络的输入和输出需要在同一空间中表示的限制。

- 在递归的部分重复利用参数，使模型保持轻量。
- 在网络中上下移动模块，以控制计算成本

这些对于构建高分辨率的修补网络是至关重要的，因为避免了前几层与后几层的计算，可以消除大部分的计算辅导。

### KCA注意力

在RFR中直接应用现有的注意设计并不是最优的，因为它们没有考虑不同的重复情况下特征图之间的一致性要求。这可能会使恢复区域的纹理模糊。所以提出了KCA 注意力，该机制可以共享重复出现之间的注意力分数，并把他们组合起来，指导patch-swap的过程。

## Work

### RFR模块



RFR模块是一种即插即用模块，具有递归推理设计，可以安装在现有网络的任何部分。可以分解为三部分:

1)区域识别模块，用于识别本次递归中要推断的区域;

2)特征推理模块，旨在对识别区域内的内容进行推理;

3)特征合并算子，用于合并中间特征图。

在该模块中，区域识别模块和特征推理模块交替往复地工作。填满洞后，将推理过程中生成的所有特征图合并，生成一个通道数固定的特征图。模型流水线如图2所示

![image-20210808121156568](C:\Users\BlackFriday\AppData\Roaming\Typora\typora-user-images\image-20210808121156568.png)

`Architecture:`

<img src="C:\Users\BlackFriday\AppData\Roaming\Typora\typora-user-images\image-20210808205308672.png" alt="image-20210808205308672" style="zoom:50%;" />

- AreaIden:区域识别
- FeatReason：特征恢复
- FeatMerge：特征合并
- IterNum:递归次数，6

#### Area Identification

用于识别每次递归中要更新的区域。

其对mask进行更新，并在卷积计算后对feature map进行重新归一化。

该部分卷积可以被表示如下：

- $F^*$通过部分卷积产生了表示feature map 。
- $f^*_{x,y,z}$​在$z^{th}$​通道x,y的位置表示特征值
- $W_z$是层中第$z^{th}$个卷积核。
- $f_{x,y}$​与$m_{x,y}$​是输入的feature patch 和mask patch（size与卷积核相同）,分别以x,y为中心。

下面为部分卷积在特征图上的计算公式：
$$
f^*_{x,y,z}=\left\{{
\begin{array}{**lr**}
W^T_z(f_{x,y}⊙m_{x,y}\frac{sum(1)}{sum(m_{x,y})})+b,&if\ sum(m_{x,y})！=0\\
0,&else\qquad\qquad\qquad
\end{array}
}
\right.
$$
同样，图层在位置i,  j处生成的新mask值可以表示为:
$$
m^*_{x,y}=\left\{{
\begin{array}{**lr**}
1,\qquad if\ sum(m_{x,y})!=0
\\
0,\qquad else
\end{array}
}
\right.
$$
根据上面的公式，经过部分卷积后，我们就可以得到新的孔洞更小的mask

我们将更新的mask和输入的mask之间的不同作为递归中要推断的区域。更新后的mask上的洞，在下一次收缩前都被保留。

feature map 经过部分卷积后，再经过归一化和激活函数处理，便输入特征推理模块。

#### feature reasoning特征推理

为了使后续的推理产生更好的结果，所以该模块要尽可能的填充准确的图像。

然而，我们这里不按常理的简单堆叠编码层与解码层，然后通过跳跃连接彼此。

推断出特征值后，更新后的掩码和部分推断的特征映射直接发送到下一个递归，无需进一步处理。



#### feature merging特征合并

递归结束，如果直接使用最后一个特征图去产生输出，会出现梯度消失，早期的迭代信息会被破坏。

我们合并了中间的特征图来解决这个问题。

然而，如果使用了卷积运算来做，会限制递归的次数，因为concat起来的通道数是固定的。直接吧所有的特征图相加会移除图像的细节，因为不同特征图上的孔洞区域不同，突出的信号被平滑。

因此我们使用自适应合并的方法来解决这个问题。输出特征图的值只从对应位置已被填满的特征图上计算。

- $F^i$为特征推理模块产生的第$i^{th}$特征图

- $f_{x,y,z}$作为在特征图F中位置为$x,y,z$的值

- $m^i$​是特征图$F^i$​的mask

- 其中的N是输入特征图的数量

  输出$\overline f$被如下定义：
  $$
  \overline f_{x,y,z}=\frac{\sum^N_{i=1}f^i_{x,y,z}}{\sum^N_{i=1}m^i_{x,y,z}}
  $$
  任意数量的特征图可以用这种方式合并，给了网络修复更大洞的可能。

  <hr wight=50% color=blue>

  ### Knowledge Consistent Attention

  在图像绘制中，使用注意模块来合成质量更好的[29]特征：在背景中搜索可能的纹理，并使用它们来替换洞中的纹理。但是现有的注意力不理想，因为在不同的循环中patch交换过程是独立执行的。合成的特征图的各组成部分之间的差异可能会在合并时损坏特征图。

  于是提出了一个新的KCA注意力。它不同以往的注意力机制，它的注意分数是独立计算的。KCA分数是由以前重复出现的分数按比例累加而成，一次来控制注意力之间的不确定性。

  ![image-20210808203223697](C:\Users\BlackFriday\AppData\Roaming\Typora\typora-user-images\image-20210808203223697.png)

  第$i^{th}$个迭代的输入特征图表示为$F^{i}$​.一开始，我们测量每对特征像素之间的余弦相似度。
  $$
  \hat{sim}^i_{x,y,x',y'}=<\frac{f_{x,y}}{||f_{x,y}||},\frac{f_{x',y'}}{||f_{x',y'}||}>
  $$
  $\hat{sim}^i_{x,y,x',y'}$​表示了在位置$(x,y),(x',y')$之间的相似度

  然后我们通过平均目标像素的像素度，来平滑相邻区域的像素的注意力分数。
  $$
  \hat{sim'}^i_{x,y,x',y'}=\frac{\sum_{p,q∈\{-k,....,k\}}\hat{sim}^i_{x,y,x',y'}}{k×k}
  $$
  

  然后，我们使用softmax函数生成位置(x, y)像素的分量比例。生成的分数图被表示为$score '$

  为了计算一个像素的最终注意力得分，我们首先决定是否要参考该像素在之前的递归中的分数。给定一个被认为有效的像素（mask的值为1），它在当前递归中的注意分数将分别计算为：

  *当前递归和之前递归的原始分数和最终分数的加权和。*

  形式上，如果位置(x, y)的像素是上次递归中的有效像素，我们自适应地将上次递归中的像素的最终得分与该递归中计算的得分结合起来，如下所示:
  $$
  score^i_{x,y,x',y'}=λscore’^{\ i}_{x,y,x',y'}+(1-λ)score^{i-1}_{x,y,x',y'}
  $$
  λ是一个可学习的参数

  否则，如果该像素在最后一次递归中无效，则不进行额外操作，并计算当前递归中该像素的最终注意评分如下:

$$
score^i_{x,y,x',y'}=λscore’^{\ i}_{x,y,x',y'}
$$

最后，利用注意力评分重建特征图。具体计算位置(x,  y)处的新feature map如下:
$$
\hat{f}{^i_{x,y}}=\sum_{x'∈1,……W\ y'∈1,,……H}score^i_{x,y,x',y'}f^i_{x',y'}
$$
重构特征映射后，输入特征F与重构特征映射$\hat F$concat后送到卷积层:
$$
F'^i=φ(|\hat F,F|)
$$
其中$F'$​为重构的特征图，φ为像素卷积。

## Architecture



我们在RFR模块之前和之后分别放置2和4个卷积层。为了简化训练，我们人为地选择递归数IterNum为6。

KCA模块位于RFR模块特征推理模块的第三层之后。对于图像生成学习，使用预先训练和固定的VGG-16的感知觉损失和风格损失。感知损失和风格损失比较生成图像的深度特征图与地面真实之间的差异。这种损失函数可以有效地向模型传授图像的结构和纹理信息。

这些损失函数的形式如下：
$$
L_{perceptual}=\sum^N_{i=1}\frac{1}{H_iW_iC_i}|φ^{gt}_{pool_i}-φ^{pred}_{pool_i}|_1
$$

- $φ_{pool_i}$​表示VGG-16中第$i^{th}$​个池化层。

- $H_iW_iC_i$表示第$i^{th}$特征图中的高，宽，通道

  样式损失的计算如下:
  $$
  φ^{style}_{pool_i}=φ_{pool_i}φ^{T}_{pool_i}
  $$
  

$$
L_{style}=\sum^N_{i=1}\frac{1}{C_i×C_i}|\frac{1}{H_iW_iC_i}φ^{style_{gt}}_{pool_i}-φ^{style_{pred}}_{pool_i}|_1
$$

此外，我们的模型中还使用了分别计算unmask区域和mask区域L1差异的$L_{valid}，L_{hole}$。

综上，我们的总损失函数为:
$$
L_{total}=λ_{hole}L_{hole}+λ_{valid}L_{valid}+λ_{style}L_{style}+λ_{perceptual}L_{perceptual}
$$
这种损失函数组合由于更新的参数数量较少，也使得训练更加高效。












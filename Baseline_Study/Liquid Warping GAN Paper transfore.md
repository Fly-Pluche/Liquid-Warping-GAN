# Abstract

We tackle the human motion imitation, appearance transfer , and novel view  synthesis within a unified framework, which means that the model once being  trained can be used to handle all these tasks. The existing taskspecific methods  mainly use 2D keypoints (pose) to estimate the human body structure. However,  they only expresses the position information with no abilities to characterize  the personalized shape of the individual person and model the limbs rotations.  In this paper, we propose to use a 3D body mesh recovery module to disentangle  the pose and shape, which can not only model the joint location and rotation but  also characterize the personalized body shape. To preserve the source  information, such as texture, style, color, and face identity, we propose a  Liquid Warping GAN with Liquid Warping Block (LWB) that propagates the source  information in both image and feature spaces, and synthesizes an image with  respect to the reference. Specifically, the source features are extracted by a  denoising convolutional auto-encoder for characterizing the source identity  well. Furthermore, our proposed method is able to support a more flexible  warping from multiple sources. In addition, we build a new dataset, namely  Impersonator (iPER) dataset, for the evaluation of human motion imitation,  appearance transfer, and novel view synthesis. Extensive experiments demonstrate  the effectiveness of our method in several aspects, such as robustness in  occlusion case and preserving face identity, shape consistency and clothes  details. All codes and datasets are available on  https://svip-lab.github.io/project/impersonator.html.

## 提出原因

现有的针对任务的方法主要是利用二维关键点(位姿)来估计人体结构。但是，它们只能表达位置信息，不能描述个体的个性化形状和建模肢体旋转。

## 优点/创新

- 在一个网络结构中可以进行人体动作模仿，外观迁移，新视图合成
- 一种基于三维体网格恢复模块的姿态和形状解耦方法，该模块不仅可以对关节的位置和旋转进行建模，还可以对个性化的身体形状进行表征。
-  Liquid Warping Block (LWB)：可以保留源信息，如纹理、风格、颜色和人脸身份，在图像和特征空间中传播源信息，并根据参考合成图像。

## 数据集

Impersonator  (iPER)数据集，用于人体运动模仿、外观迁移和新视图合成的评估。大量的实验证明了该方法在遮挡情况下的鲁棒性和保持人脸身份、形状一致性和衣服细节方面的有效性。

## 介绍

下图很好的展示了本网络的三大用处，人体动作模仿，外观迁移，新视图合成。在再现场景、角色动画、虚拟衣服试穿、电影或游戏制作等方面具有巨大的应用潜力。

![image-20210613205639290](https://gitee.com/Black_Friday/blog/raw/master/image/image-20210613205639290.png)

### 现有的人的图像合成方法

![image-20210613210425187](https://gitee.com/Black_Friday/blog/raw/master/image/image-20210613210425187.png)

#### 连接

工作：导致图像模糊，失去源身份。

缺点：导致图像模糊，失去源身份。

如图2  (a)所示，将源图像(具有其姿态条件)和目标姿态条件进行连接，然后将源图像(具有其姿态条件)和目标姿态条件输入经过对抗训练的网络，生成具有所需姿态的图像。但是，直接连接没有考虑到空间布局，而且生成器将源图像的像素放置到正确的位置是模糊的。而且对生成器来说，原图像的像素对应的正确位置是模糊的。

#### 纹理扭曲

受空间变压器网络(STN)[10]的启发，我们提出了一种纹理扭曲方法[1]，如图2  (b)所示。该方法首先根据源姿态和参考姿态拟合一个粗糙的**仿射变换矩阵**，利用STN将源图像扭曲成参考姿态，并根据扭曲后的图像生成最终结。然而，纹理扭曲并不能保留源信息，比如颜色、风格或人脸身份，因为在进行了几次降采样操作(如stride卷积和pooling)后，生成器可能会丢失源信。

#### 特征扭曲

当代作品[4,31]提出将源图像的深度特征扭曲成目标姿态，而不是图像空间的特征，如图2 (c)所示，称为feature  warp。然而，编码器在特征扭曲中提取的特征不能保证准确地表征源身份，从而不可避免地产生模糊或低保真图像。

### 现有方法的缺陷

上面三种现有的方法在生成非真实感图像方面遇到挑战，原因如下：

1. 服装在质地、风格、色彩等方面的多样性，以及高结构的人脸身份难以在其网络架构中捕捉和保存;
2. 铰接式和可变形的人体导致大的空间布局和几何变化的任意姿态操作;
3. 所有这些方法都不能处理多源输入，如在外观传输中，不同的部分可能来自不同的source people。

## Liquid warp Block（LWB）

用于解决信息丢失问题

1. 利用去噪卷积自编码器提取保留源信息的有用特征，包括纹理、颜色、风格和人脸身份;
2. 将各局部特征融合成全局特征流，进一步保留源细节;
3. 支持多源扭曲，如在外观传输中，将一个源的头部特征和另一个源的身体特征进行扭曲，并聚合成一个全局特征流。这将进一步增强每个来源部分的本地身份。

## 三维体网格

现有的方法主要依赖2D姿态[1,19,31]、密集姿态[22]和体解析[4]。这些方法只考虑布局位置，忽略了个性化的形状和肢体(关节)旋转，然而这在人体图像合成中比布局位置更为重要。

例如，在一个极端的情况下，一个高个子模仿一个矮个子的动作，使用二维骨骼、密集的姿态和身体解析条件，不可避免地会改变高个子的身高和体型，如图6底部所示。为了克服它们的缺点，我们使用了一个参数统计人体模型，SMPL[2,18,12]模型，它将人体分解为姿态(关节旋转)和形状。它输出的是三维网格(没有衣服)，而不是关节和部件的布局。此外，通过匹配两个三维三角网格之间的对应关系，可以很容易地计算变换流，这比以前的关键点拟合仿射矩阵更准确，误差更少[1,31]。![image-20210613214242399](https://gitee.com/Black_Friday/blog/raw/master/image/image-20210613214242399.png)

# 思想

Liquid Warping GAN 包括三个阶段，body mesh recovery, flow composition and a GAN module with Liquid Warping Block  (LWB)。

![image-20210613215526369](https://gitee.com/Black_Friday/blog/raw/master/image/image-20210613215526369.png)

## Body Mesh Recovery Module

如图3（a）中所示，$I_s$是 source image ，$I_r$是reference image 。这个阶段是用于预测每张图像的运动学姿态（姿态旋转），形状参数以及三维网格。

我们使用HMR作为三维姿态和形状估计器，因为它能很好的兼顾精度和效率。

`HMR`

图像首先被ResNet-50编码成一个$R^{2048}$的特征，然后紧接着是使用迭代的3D回归网络预测θ∈$R^{72}$和形状β∈$R^{10}$的SMPL，以及弱视角相机K∈$R^3$。SMPL是一个三维体模型可以被定义为一个可微函数M（θ，β)∈$R^{N_v×3}$，他通过$N_v$= 6890个顶点和$N_f$= 13776个面参数化一个三角网格，姿态参数θ∈$R^{72}$和β∈$R^{10}$。

形状参数β：是一个低维形状空间的系数从成千上万的注册扫描中学习。

姿态参数θ：是通过正运动学连接骨骼的关节旋转（the pose parameters θ are the joint rotations that articulate the bones via  forward kinematics.）。

通过该过程，分别得到源图像的体重构参数{$K_s,θ_s,β_s, M_s$}和参考图像的体重构参数 

{$K_r,θ_r,β_r,M_r$}。

## Flow Composition Module

这个流组成模块首先根据两个对应映射及其在图像空间的projected vertices来计算transformation flow T，然后它会从source image  $I_s$中分离出masked background $I_{bg}$，最后它会根据transformation flow T扭曲source image和产生一个warp image $I_{syn}$

## Liquid Warping GAN

在前面的基础上，我们首先在照相机视角$K_s$下绘制渲染源网格$M_s$和参考网格$K_s$的对应图，这里，我们将源和参考对应映射分别表示为$C_s$和$C_t$。在本文中，我们使用了一个全微分渲染器，神经网格渲染器(NMR)[13]。因此我们通过弱透视相机将source $V_s$中的project vertices 投影到二维图像空间中。然后我们计算每个网格面的重心坐标为$f_s$∈$R^{N_f×2}$。接下来，我们会通过匹配source 对应的映射与其网格面坐标$f_s$，和参考对应映射之间的对应进行，来计算transformation flow T∈$R^{H×W×2}$。

这里H×W是图像的大小。前景图$I_{ft}$和蒙版背景图$I_{bg}$是源于源图$I_s$，基于$C_s$  。最后，我们通过变换流T对源图像$I_{s}$进行扭曲，得到扭曲后的图像$I_{syn}$，如图3所示。

![image-20210613215526369](https://gitee.com/Black_Friday/blog/raw/master/image/image-20210613215526369.png)

## Liquid Warping GAN

这一阶段在所需条件下合成高保真的人体图像。

1. 合成图像背景
2. 基于可见部分的不可见预测
3. 通过重构SMPL生成衣服，头发等像素

### 生成

我们的生成器有三个部分

- $G_{BG}$：基于将masked background image $I_{bg}$和在颜色通道上对$C_s$进行二值化得到的mask串联，去生成现实的背景图像$\hat{I}_{bg}$(图3（c）中的第一个分流)
- $G_{SID}$：是一个去噪的自卷积目的是指导编码器提取能够保存源信息的特征。跟$\hat{I}_{bg}$类似，它将masked source foreground $f_{ft}$和网格图$C_s$（一共六个通道）作为输入，并且重新构造了source front image $\hat{I}_{bg}$
- $G_{TSF}$：它将warp foreground 通过双线性采样和网格图$C_t$（六个通道）作为输入，为了获得纹理，风格，颜色等源图信息，我们构造了一个新的Liquid Warping Block（LWB），用于连接带有目标流（target）的source，它将协调来自$G_{SID}$的source features，并把它们融入transfer stream $G_{TSF}$

<img src="C:\Users\BlackFriday\AppData\Roaming\Typora\typora-user-images\image-20210614212805217.png" alt="image-20210614212805217" style="zoom: 67%;" />

它解决了多种来源，如在人体外观转移中，保留来源一的头部，从来源二穿上外层衣服，而从来源三穿下外层衣服。不同部分的特征在进入$G_{TSF}$时，通过自己的转化流独立的聚合。

下面举个栗子（如下图 图4）：

$X_{S_1}^l$跟 $X_{S_2}^l$是特征图在第$l^{th}$层的通过$G_{TSF}$提取的，每一部分的特征图是通过他们扭曲他们自己的转化流和$G_{TSF}$的特征聚合。我们使用双线性采样（BS）去扭曲source features $X_{S_1}^l$跟 $X_{S_2}^l$，带有转化流$T_1$和$T_2$的各自区别。最后的输出特征包含如下：
$$
\widehat{X^l_t}=BS (X^l_{S_1},T_1)+BS (X^l_{S_2},T_2)+X^l_t
$$
`PS`

我们只采用了两个source作为例子，实际上可以轻松扩展为多个source



<img src="C:\Users\BlackFriday\AppData\Roaming\Typora\typora-user-images\image-20210614210055825.png" alt="image-20210614210055825" style="zoom: 80%;" />

## 一些细节

- 这三个模块都有相似的ResUnet结构（即没有共享参数的ResNet + U-Net）

- 图3中的A 与P 分别是attention map与color map<img src="C:\Users\BlackFriday\AppData\Roaming\Typora\typora-user-images\image-20210614222810602.png" alt="image-20210614222810602" style="zoom:25%;" />

- 鉴别器：遵循Pix2Pix[9]的架构。

- 最后图像的计算公式：
  $$
  \hat{I}_S=P_S*A_S+\hat{I}_{bg}*(1-A_S)\\
  \hat{I}_t=P_t*A_t+\hat{I}_{bg}*(1-A_t)
  $$
  
- 

- 身体恢复模块，我们按照HMR的损失函数与网络架构。这里，我们使用了HMR的预训练模型

- $I_s与I_r$：从每个视频中随机采样一对图片，一个为$I_s$，另一个为$I_r$。

- 论文的mini-batch为4

- 优化器为Adam

  

## 损失函数

损失函数是由四个部分构成，分别是perceptual loss, face identity loss, attention regularization loss, 和 adversarial loss.

对于生成器，完整的目标函数如下，其中的$λ_p,λ_f,λ_a$是the weights of perceptual, face identity and attention losses.论文中将它们设置为10.0，5.0，1.0
$$
L^G=λ_pL_p+λ_fL_f+λ_aL_a+L^G_{adv}
$$
对于鉴别器
$$
L^D=\sum[D(\hat{I}_t,C_t)+1]^2+\sum[D(I_r,C_t)-1]^2
$$

### Perceptual Loss

在VGG子空间下，它对于重构source image $\hat{I}_s$和生成更接近ground truth  $I_s$和$I_r$的target image $\hat{I}_t$
$$
L_p=||f(\hat{I}_s)-f(I_s)||_1+||f(\hat{I}_t)-f(I_r)||_1
$$
其中的f是VGG-19的预训练模型

### Face Identity Loss

将合成的目标target image $\hat{I}_t$规范化，使其余ground truth $I_r$，从而推动生成器去保持脸的身份识别
$$
L_f=||g（\hat{I}_t)-f(I_r)||
$$
g是预训练的SphereFaceNet

### Adversarial Loss

它推动了合成转化图像的分布转化为真实图像的分布。我们使用一个类似PatchGAN中生成LSGAN-110的损失——$LSGAN_{110}$  。鉴别器D规范化$I_t$,使看起来更加真实，我们使用条件判别器，它以生成的图像和对应的映射$C_6$(6个通道)作为输入。
$$
L^G_{adv}=\sum{D}(\hat{I}_t,C_t)^2
$$

### Attention Regularization Loss

它为规范注意力图A，使其变得平滑并且防止他们饱和。考虑到这里的注意力图A和颜色图P不是ground truth ，他们从上述的损失结果梯度中学习。然而，这些注意力掩码容易饱和为1，阻止了生成器的工作。为了防止这一情况，我们将mask规范化，使它更接近3D身体网格渲染的轮廓。因为剪影是一个粗糙的图，它包含了没有衣服与头发的身体mask，我们还在A上执行了一个像[25]这样的全变化规范（Total Variation Regularization）以弥补剪影的缺失，并结合预测背景的像素$\hat{I}_{bg}$和颜色图P，进一步增强平滑的空间色彩。
$$
TV(A)=\sum[A(i,j)-A(i-1,j)]^2+[A(i,j)-A(i,j-1)]^2
$$

$$
L_a=||A_s-S_s||^2_2+||A_t-S_t||_2^2+TV(A_s)+TV(A_t)
$$



### 小问号

为什么用L1范数

因为L2范数是计算预测像素与真实像素之间的欧几里得距离，会产生模糊的结果。因为欧式距离是通过平均所有可能的输出而最小化的。

# 相关

![image-20210615103108386](C:\Users\BlackFriday\AppData\Roaming\Typora\typora-user-images\image-20210615103108386.png)

图5是对不同任务的不同实现。

Motion Imitation：我们首先将位置参数$θ_r$复制到原位置参数中，得到SMPL的综合参数和3D网格，$M_t=M(θ_r,β_s)$。接下来在摄像机视角$K_s$下，我们渲染source mesh $M_s$的通信图和综合网格$M_t$,

# 总结

该方法基于SMPL模型和液体扭曲块(Liquid warp Block,  LWB)模型，可以扩展到人的外观转移和新视图合成等任务，一个模型就可以完成这三个任务。

- 提出了一个LWB来传播和解决源信息，如纹理、风格、颜色和人脸身份在图像和特征空间的丢失;
- 结合LWB和三维参数化模型，为人体运动模拟、外观传递和新视图合成提供了统一的框架;
- 针对这些任务，特别是视频中的人体运动模仿，我们建立了数据集，并发布了所有代码和数据集，以方便社区进一步研究。

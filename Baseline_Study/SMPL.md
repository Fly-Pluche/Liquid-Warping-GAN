



# SMPL: A Skinned Multi-Person Linear Model论文解读

论文：https://files.is.tue.mpg.de/black/papers/SMPL2015.pdf

## 相关知识（动画制作专业术语）

- vertex（顶点）：每个动画动画模型可以看成多个三角形（四边形）组成，每个三角形可以看成一个顶点。顶点越多，动画模型越精细（感觉可以理解为图片中的像素，像素越高图片质量越好）。

- vertex weights（顶点权重）：用于变形网格mesh。

- 骨骼点：人体的一些关节点，类似于人体姿态估计的关键点。每个骨骼点都由一个三元组作为参数去控制（可以看欧拉角，四元数相关的概念）

- 骨骼蒙皮（Rig）：建立骨骼点与顶点的关联关系。每个骨骼点会关联许多顶点，并且每个顶点权重不一样。通过这种关联关系，就可以通过控制**骨骼点的旋转向量**来控制整个人的运动。

- BlendShape：另一种控制角色运动的方法。用于从笑脸到正常脸这类的平滑过度动画。因为不定义骨骼点，所以相比Rig比较方便。

- 蒙皮：将一个姿态转变为另一个姿态，使用的转换矩阵叫做蒙皮矩阵（Linear Blend Skinning算法）。

  ### 较无关的知识点

- 纹理贴图：动画人体模型表面的纹理，即裤子衣服。
- uv map：将3D多边形网格展开到2D平面，得到UV图像。
- texture map：将3D多边形网格表面的纹理展开到2D平面，得到纹理图像。
- 拓扑（topology）：重新拓扑是将高分辨率模型转换为可用于动画的较小模型的过程。

两个mesh拓扑结构相同是指两个mesh上面任一个三角面片的三个顶点的ID是一样的（如某一个三角面片三个顶点是2,5,8；另一个mesh上必有一个2,5,8组成的三角面片）

## 输入模型的参数

$\vecβ和\vecθ$：是SMPL的输入，可以预先从HumanModelRecovery中得到。

$\vec{\beta}:$shape parameters(10维)

$\vec{θ}:$Pose parameters（3K维，K是骨架节点数）

$\vec{\omega}:$Scaled axis of rotation,the3 pose parameters corresponding to a particular joint.(旋转的缩放轴，对应于特定关节的3个姿态参数。)

$\vec{θ^*}:$Zero pose or rest pose；the effect of the pose blend shapes is zero for that pose(该姿态的姿势混合形状的效果为零).

## 要通过训练集训练获取的参数：

{ $\overline{T},W_1,S,J,P$ }

$\overline{T}∈R^{3N}:$由N个串联的顶点表示的初始状态下的平均模型，$\vec{θ^*}$

$W_1∈R^{N×K}：$: LBS/QBS混合权重矩阵，即关节点对顶点的影响权重 (第几个顶点受哪些关节点的影响且权重分别为多少)

$S=[S_1,……,S_{|\vec{\beta}|}]∈R^{3N×|\vec\beta|}:$为形状位移矩阵 (形状位移的标准正交主成分)

$P=[P_1,……，P_{9R}]∈R^{3N×9R}$:所有207个姿势混合形状组成的矩阵 (由姿势引起位移的正交主成分)

$J:$将rest vertices转换成rest joints 的矩阵（获取T pose的关节点坐标的矩阵）[完成顶点到关节的转化]

## SMPL的输出

N vertices(顶点)：6890

## 参数

- $\vecβ$：代码人体高矮胖瘦、头身比等10个ShapeBlendPose参数
- $\vecθ$：代表人体整体运动位置姿态和24个关节的相对角度。一共有75个参数

75=24*3+3（每个关节3个自由度，再加上3个根节点）

- $M$：SMPL function
- $W$：Skinning function
- $B_P$：Pose blendshapes function
- $B_s$：Shape blendshapes function
- $J$：Joint regressor,Predicts joints from surface
- $K:$骨架节点数
- $M(\vec{β},\vec{θ})=W(T_p(\vec{β},\vec{θ}),\vec{θ},W_1)$![image-20210624225045815](C:\Users\BlackFriday\AppData\Roaming\Typora\typora-user-images\image-20210624225045815.png)

$R^{3N×3K×|θ|×|W_1|}\longrightarrow{R^{3R}}$(the standard linear blend skinning function):

从模板SMPL模型中取vertics$\overline{T}$,joint locations $J$,a pose $\vec{θ}$ and the blend weight W ,输出posed vertices。

1. $J(\vec{β}):R^{|\vec{β}|}\rightarrow R^{|3K|}$： a function to predict K join locations.

2. $T_p(\vec{β},\vec{θ})=T+B_s(\vec{β})+B_p(\vec{θ})$

3. $B_s(\vec{\beta}):R^{|\vec{β}|}\rightarrow R^{|3K|}$：a blend shape function 

   input：shape parameters $\vec{\beta}$

   output：a blend shape sculpting the subject identity(塑造主体身份的混合形状)

4. $B_s(\vec{θ}):R^{|\vec{θ}|}\rightarrow R^{|3K|}$: a pose-dependent blend shape function .考虑姿态相关变形影响



## Blend skinning



 



## 补充

$\vec\beta$:10个shape参数分别对应的物理意义：（实际有50个参数，开源的只有10个）smpl官网的unity模型可以用slider 控制参数变化



0. 代表整个人体的胖瘦和大小，初始为0的情况下，正数变瘦小，负数变大胖（±5）

1. 侧面压缩拉伸，正数压缩
2. 正数变胖大
3. 负数肚子变大很多，人体缩小
4. 代表 chest、hip、abdomen的大小，初始为0的情况下，正数变大，负数变小（±5）
5. 负数表示大肚子+整体变瘦
6. 正数表示肚子变得特别大的情况下，其他部位非常瘦小
7. 正数表示身体被纵向挤压
8. 正数表示横向表胖
9. 正数表示肩膀变宽

> 参考：
>
> 1. [SMPL: A Skinned Multi-Person Linear Model论文解读](https://blog.csdn.net/JerryZhang__/article/details/103478265?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522162452560716780271525318%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=162452560716780271525318&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_v2~rank_v29-6-103478265.first_rank_v2_pc_rank_v29&utm_term=smpl%E8%AE%BA%E6%96%87)
> 2. https://zhuanlan.zhihu.com/p/256358005


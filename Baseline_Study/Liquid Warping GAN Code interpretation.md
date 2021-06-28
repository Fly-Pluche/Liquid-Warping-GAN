

默认的Model=impersonator_trainer

这个工程文件中的代码有点复杂，我们先来挨个看看…..

# train.py

### impersonator_trainer:

![](C:\Users\BlackFriday\AppData\Roaming\Typora\typora-user-images\image-20210620212915727.png)

# HumanModelRecovery

> 路径：./networks/hmr.py    255行

#### 作用

返回一个包含beta跟theta的参数betas（beta与theta将会输入接下来的SMPLRenderer）

#### 构成

- preACRResNet50
- SMPL（这个SMPL不同于后面的SMPLRenderer）

输入顺序：

preACRResNet50$\longrightarrow{feature}\longrightarrow$ThetaRegressor

## preACRResNet50

```python
def preActResNet50():
    return PreActResNet(PreActBottleneck, [3, 4, 6, 3])
```

我们来看一下PreActResNet的forward部分

$conv\longrightarrow{Maxpool}\longrightarrow{laver_{1,2,3,4}}\longrightarrow{relu}$

```python
class PreActResNet():
    def forward(self, x):
        out = self.conv1(x)
        out = F.max_pool2d(out, kernel_size=3, stride=2, ceil_mode=True)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.post_bn(out))

        # need global avg_pooling
        out = F.avg_pool2d(out, 7)

        out = out.view(out.size(0), -1)
        return out
```

layer主要是由PreActBottleneck堆叠而成,而PreActBottleneck主要就是relu,bn,conv

## ThetaRegressor

### 输出

```python
:return: a list contains [[theta1, theta1, ..., theta1],
[theta2, theta2, ..., theta2], ... , ],
        shape is iterations X N X 85(or other theta count)
```

### 过程

将feature循环丢入全连接层提取特征，将theta自加迭代

```python
for _ in range(self.iterations):
    total_inputs = torch.cat([x, theta], dim=1)
    theta = theta + self.fc_blocks(total_inputs)
```

## 后面会用到的函数

### get_details

```python
inputs:
    theta: N X (3 + 72 + 10)

return:
    thetas, verts, j2d, j3d, Rs
```





# SMPLRender



#### 初始化部分：

- 创造了一个network（self._init_create_networks()）
- 定义训练的variable 与loss
- 加载优化器与网络
- 预取变量

#### self._init_create_networks()

- 身体恢复流（BodyRecoveryFlow）
- 生成器（self._create_generator()）
- 判别器（self._create_discriminator()）

#### class BodyRecoveryFlow

- 人体多模态恢复（HumanModelRecovery）
- SMPLRenderer

#### HumanModelRecovery

![image-20210620211735278](C:\Users\BlackFriday\AppData\Roaming\Typora\typora-user-images\image-20210620211735278.png)

加载数据

#### _create_render

![image-20210620211819264](C:\Users\BlackFriday\AppData\Roaming\Typora\typora-user-images\image-20210620211819264.png)

估计也是加载数据

#### 生成器 判别器

![image-20210620212544523](C:\Users\BlackFriday\AppData\Roaming\Typora\typora-user-images\image-20210620212544523.png)

## run_imitator.py 中的代码

主要是将所有的实现函数封装在Imitator这个类中

```python
imitator = Imitator(test_opt)
```

Imitator主要初始化了

- bgnet
- hmr
- render
- pre-processor

最后再使用Imitator中的personalize与inference


## Lecture 8：Adversarial Attack

> Lectured by HUNG-YI LEE (李宏毅)
>
> Recorded by Yusheng zhao（yszhao0717@gmail.com）

----------------

[TOC]

------

> MOTIVATION：
>
> 我们所训练的各式各样的神经网络如果想落地应用（deployed），不仅需要正确率高，还需要**应付来自人类的恶意攻击**。
>
> 很多network实际上主要作用就是侦测来自人类的恶意攻击，譬如垃圾邮件甄别

### 人类的恶意是什么样子的：How to Attack？

E.g. 图像识别的系统

将输入图片加入小小的噪音（肉眼没法看出来），被称之为Attacked Image，

预期攻击效果有两种：其一是仅让其识别错误（Non-Targeted）；其二是不仅让其识别错误，还要使其结果分类到预期的类别（Targeted）…

<img src="https://s1.328888.xyz/2022/05/04/hWhg3.png" alt="image-20220415080008405" style="zoom:67%;" />

以一个ResNet-50的Network 为例

<img src="https://s1.328888.xyz/2022/05/04/hW454.png" alt="image-20220415080220855" style="zoom: 50%;" />

这个0.64和1是置信度分数。攻击成功而杂讯肉眼无法分辨。将这两张图片相减并放大差距，杂讯如右上角所示，两张照片确实不一样

<img src="https://s1.328888.xyz/2022/05/04/hWHaB.png" alt="image-20220415080426139" style="zoom: 50%;" />

事实上，我们可以把这只猫加上杂讯，让去变为任何其他东西。例如键盘（Keyboard）

有趣的是，当我们在原图片加上适度的肉眼可分辨的杂讯时，可能分类器的结果并不会被误解。即便犯错，似乎也有“有尊严”的解释。

<img src="https://s1.328888.xyz/2022/05/04/hWDQT.png" alt="image-20220415080721552" style="zoom: 50%;" />

#### How to Atatck

对于Network $f$（参数是固定的）输入一张影像姑且称之为$x^0$，输出是一个distribution，称之为$y^0$。那么$y^0=f(x^0)$

- 如果攻击目标是non-targeted的。我们要找到一张新的图片$x$，丢进Network中，输出$y$，要求其和ground Truth $\hat{y}$的差距越大越好，就算是攻击成功。如上过程实际上要求解一个Optimization的问题，定义我们的损失函数$L(x) = -e(y,\hat{y})$，通常是两者的交叉熵的负值。目标函数为：
  $$
  x^* = arg \ min \ L(x)
  $$

- 如果攻击目标是targeted的。我们需要预先设定好我们的目标称之为$y^{target}$——实际上是一个独热向量。我们找到一张新的图片$x$，丢进Network中，输出$y$，最后希望$y$不仅和$\hat{y}$越远越好，而且要和$y^{target}$越近越好。这时候我们的损失函数为$L(x) = -e(y,\hat{y}) + e(y,y^{target})$，$e(·)$求交叉熵。

- 攻击效果要求杂讯是肉眼无法辨别的，那么对于输入$x$要和原图$x^0$越接近越好，此时目标函数为
  $$
  x^* = arg \ \underset{d(x^0,x)\leq \epsilon}{min} \ L(x)
  $$
  这里$d(x^0,x)\leq \epsilon$的阈值由人类感知的极限所决定。

#### 计算Non-perceiveable

假设<img src="https://s1.328888.xyz/2022/05/04/hWdH2.png" alt="image-20220415082606475" style="zoom:50%;" />

- L2-Norm：$d(x^0,x) = ||\Delta x_i||_2 = \sum||\Delta x_i||^2$
- L-infinity：$d(x^0,x) = ||\Delta x_i||_\infty\ = max\{|\Delta x_1|,|\Delta x_2|,|\Delta x_3|,...\}$

也有其他方法来计算距离，但是我们在计算中必须考虑到人类感知的情形。举例说明可能L-infinity也许更符合实际的需求。

$x$和$x^0$的距离衡量方式必须根据Domain Knowledge，或者说具体问题具体分析。对于一个图像分类系统可能如上情况所述，但是对于语音辨识系统，我们需要找出语音中人类比较不敏感的element，距离衡量方式随之产生变化。

#### 攻击方法

我们现在有
$$
x^* = arg \ \underset{d(x^0,x)\leq \epsilon}{min} \ L(x)
$$
我们只需要把网络的input看作是网络的一部分，和一般训练网络一样，通过Gradient Descent来minimize我们的损失函数，对输入的$x$进行调整。

- 把$x$初始化为$x^0$（从$x^0$开始找）

- 迭代的更新参数$For \ t =1 \ to \ T$，在每一个迭代$t$里边，我们都会计算梯度.由$\large g = [\frac{\part L}{\part x_1}|_{x = x^{t-1}},\frac{\part L}{\part x_2}|_{x = x^{t-1}},..]^T$于network的参数是fixed的，所以特别的这里的梯度不是参数对loss的梯度，而是input的图片$x$对loss的梯度（gradient），迭代如下$x^t \leftarrow x^{t-1} - \eta g$

- 加入限制：$ d(x^0,x)\leq \epsilon$，如果更新完$x$发现限制不满足，那就更改$x$使其满足限制（下图以L-infinity为例）
  $$
  For \ t = 1 \ to \ T: \ x^t  = x^{t-1} - \eta g
  	\\ IF \ d(x^0,x^t) \gt \epsilon \ THEN \ x^t \leftarrow fix(x^t) 
  $$
  <img src="https://s1.328888.xyz/2022/05/04/hWlcM.png" alt="image-20220415110810986" style="zoom:50%;" />

  只要update的超出了框框（蓝点），那就把它fix会框内最近的点。

不同的攻击手段：采用不同的constraint或者不同的optimization方法；但是通常都用梯度下降法。

<img src="https://s1.328888.xyz/2022/05/04/hWtS7.png" alt="image-20220415114841757" style="zoom:67%;" />

##### [FGSM（Fast Gradient Sign Method）](https://arxiv.org/abs/1412.6572)

>如同埼玉老师：一发命中

一击必杀——一个update就找出可以attaack成功的Image，步骤如下

- 把$x$初始化为$x^0$（从$x^0$开始找）

- 只迭代一次，$x^t \leftarrow x^{t-1} - \eta g$，这里的学习率$\eta$就直接等于constraints的阈值$\epsilon$

- 梯度的设计
  $$
   g = [sign(\frac{\part L}{\part x_1}|_{x = x^{t-1}}),sign(\frac{\part L}{\part x_2}|_{x = x^{t-1}}),..]^T
  $$
  $sign(·)$即符号函数，值为+1或-1 

- 效果：（L-infinity作为距离衡量方法），一次攻击后，所得到的$x$一定落在以$x^0$为正中心的四个方框角上（所以最后的调整就只有四个选择，向上向下向左向右）

  <img src="https://s1.328888.xyz/2022/05/04/hWWrX.png" alt="image-20220415115928700" style="zoom:67%;" />

##### 改进版：[Iterative FGSM](https://arxiv.org/abs/1607.02533)

多跑几个迭代（作业能过medium）

<img src="https://s1.328888.xyz/2022/05/04/hWCZC.png" alt="image-20220415120700128" style="zoom:45%;" />

坏处：一不小心就出界了，所以最后还要fix下，出界的点修正到四个角中最接近的那个。

<img src="https://s1.328888.xyz/2022/05/04/hWzsZ.png" alt="image-20220415120549628" style="zoom:67%;" />

### White Box v.s. Black Box

- 白箱攻击：神经网络/模型参数已知的攻击。如上所说的攻击方式，神经网络的参数是固定的，我们训练调整输入的攻击图像$x$。事实上我们无法从大部分online API中获取模型参数。在这种情况下，如果我们不把模型参数公开，是否就能避免人为的攻击呢？

- 答案是否定的，**黑箱攻击（Black Box Attack）**依然是有可能的。

### Black Box Attack

- **Black Network**：神经网络参数未知的模型（Be Attacked）。Training data：该黑箱网络的训练资料

  用训练资料训练模仿黑箱网络的相似的一个神经网络，姑且称之为“**Proxy Network**”。如果Proxy Network和黑箱网络有一定程度的相似度的话，我们只需要对Proxy Network采用白箱攻击，所得到的攻击过的图像$x$拿到黑箱网络输入攻击也有效果。

  <img src="https://s1.328888.xyz/2022/05/04/hW09g.png" alt="image-20220416105109301" style="zoom:67%;" />

- Black Network：神经网络参数未知的模型；但是没有训练资料

  那就自己搜集资料：尝试对黑箱网络输入，获取输出，整理成对资料，将这个资料作为我们的训练资料，用其训练得到Proxy Network。

- 黑箱攻击非常容易成功。见https://arxiv.org/pdf/1611.02770.pdf

  <img src="https://s1.328888.xyz/2022/05/04/hWs51.png" alt="image-20220416105426662" style="zoom:67%;" />

  （对角线处是白箱攻击，成功率是100%：）除此之外，non-targeted比较容易做到，而targeted攻击比较难成功。

- **Ensemble Attack**

  <img src="https://s1.328888.xyz/2022/05/04/hW7kt.png" alt="image-20220416105836513" style="zoom:67%;" />

  解释：第一行为例——找到一张image，在网络ResNet-101、ResNet-50、VGG-16以及GoogLeNet上可以攻击成功的（Ensemble Attack），在五个网络上的上test的正确率分别为0 %、0%、0%、0%、0%.（对角线处是黑箱攻击：）

### 为什么攻击成功如此简单？

> still a puzzle.

老师介绍了一个可能（很多人相信）的原因，来自https://arxiv.org/pdf/1611.02770.pdf

<img src="https://s1.328888.xyz/2022/05/04/hWRUe.png" alt="image-20220416110515444" style="zoom:67%;" />

实验如上，图上的原点代表着尼莫的图片，横轴和纵轴分别表示这张图片往两个不同的方向移动。在VGG-16上面，横轴表示可以攻击成功的方向，纵轴是随机的方向。在另外的四个NN模型上。其可视化结果和VGG-16很相似（深蓝色区域代表会被辨识成功的图片的范围）

在攻击的方向上（横轴）特别窄，稍微加点噪声，往横轴方向移动，就掉出识别正确的领域。

[Adversarial Examples Are Not Bugs, They Are Features](https://arxiv.org/abs/1905.02175)——不同的文章也说明了之所以能攻击成功是因为**数据**的特征分布而非**模型**，而在不同的模型中数据分布是相似的，攻击成功的形式也非常类似。

👆只是某一个许多人认同的想法。

### One Pixel Attack

> 攻击成功所需要的噪声代价至少可以多大？——一个像素就行。https://arxiv.org/abs/1710.08864

<img src="https://s1.328888.xyz/2022/05/04/hWYHO.png" alt="image-20220416163426325" style="zoom: 80%;" />

局限性很大：攻击存在，但不是很powerful（不会错误识别到完全不一致的事物上，多少有点像）

### Universal Adversarial Attack

> https://arxiv.org/abs/1610.08401

不需要对不同的signal攻击特质化（specialized），图像识别攻击的通用化手段。

### Beyond Images——其他类型资料的被攻击

- Speech processing

  - 侦测语音是否合成

- NLP

  - Q&A：在文字上的adversarial attack，让不同的问题回答一样的答案。https://arxiv.org/abs/1908.07125

    <img src="https://s1.328888.xyz/2022/05/04/hWTmq.png" alt="image-20220416164903598" style="zoom:67%;" />

### Attack in the Physical World

> 发生在三次元世界中的Adversarial Attack

- 人脸识别攻击

  例子1：advhat：人脸识别的贴纸攻击……

  <img src="https://s1.328888.xyz/2022/05/04/hWcSP.png" alt="image-20220416165130736" style="zoom:50%;" />

  例子2：如上图，https://www.cs.cmu.edu/~sbhagava/papers/face-rec-ccs16.pdf

  三个角度考虑物理世界的攻击：

  - An attacker would need to find perturbations that generalize beyond a single image.

    真实世界需要多个角度看待问题，对于人脸识别的贴纸攻击，应当从所有角度，戴上贴纸都使得攻击成功。

  - Extreme differences between adjacent pixels in the perturbation are unlikely to be accurately captured by cameras.

    设备如摄像头解析度的局限性，不太好捕捉相邻像素本身之间的较大差异或加入扰动后的差异。

  - It is desirable to craft perturbations that are comprised mostly of colors reproducible by the printer.

    有某些颜色在计算机和真实世界中是有差异的，不推荐使用印刷后出现偏差的颜色。

- 自动驾驶中的标识牌识别攻击

  <img src="https://s1.328888.xyz/2022/05/04/hWwrm.png" alt="image-20220416180021547" style="zoom: 80%;" />

考虑到角度、远近距离的标识牌攻击，实际的贴纸比较招摇，如下是相对隐蔽的攻击方式。

<img src="https://s1.328888.xyz/2022/05/04/hW37A.png" alt="image-20220416180245319" style="zoom:50%;" />

（仔细看，数字3”鼻子“被拉长了；将限速35误识别为85：）来自https://www.mcafee.com/blogs/other-blogs/mcafee-labs/model-hacking-adas-to-pave-safer-roads-for-autonomous-vehicles/

### Adversarial Reprogramming

> https://arxiv.org/abs/1806.11146
>
> 把原影像识别系统放入寄生的僵尸？让它做它本来不想做的事情

<img src="https://s1.328888.xyz/2022/05/04/hWKZS.png" alt="image-20220416193225438" style="zoom:67%;" />

👆数方块的模型：将方块$y_{adv}$的图片嵌入杂讯中，杂讯加入相对应的图像$y$，丢进分类器里边，（ImageNet Classifier原来是识别图像）借用其功能来做到数方块的模型

### "Backdoor" in Model

> 来自文章：https://arxiv.org/abs/1804.00792 ——发现模型的后门，来自人类的另外一个恶意

<img src="https://s1.328888.xyz/2022/05/04/hWa9R.png" alt="image-20220416193644927" style="zoom:67%;" />

攻击从训练过程中就展开……在训练资料中图片是正常的而且标注是正常的，但是给模型开了一个后门（样本攻击）。这导致在测试中每次遇到此类样本的时候都会辨识错误。

这启示我们在使用公开的（open）训练集，小心其中做的手脚……

### 如何防御：被动 v.s. 主动

#### Passive Defense（被动防御）

<img src="https://s1.328888.xyz/2022/05/04/hWkWi.png" alt="image-20220416194054338" style="zoom:67%;" />

- 制作一个filter，让加了杂讯的图片（受到攻击）中attack signal的效果减弱（less harmfu），避免辨识错误。

  如何制作这样一个filter：（最简单的）稍微对图像做一个**模糊化（*Smoothing*）**。

  会让攻击成功的signal：非常特殊的，往往是攻击成功的一个方向，并不是随机sample出来的噪声。局限性：模糊化图片会让分类器对图像的置信度下降。

  <img src="https://s1.328888.xyz/2022/05/04/hWnkv.png" alt="image-20220416194536852" style="zoom: 50%;" />

- **Image Compression**：对图像压缩再解压缩

  > 来自文章https://arxiv.org/abs/1704.01155以及https://arxiv.org/abs/1802.06816

  <img src="https://s1.328888.xyz/2022/05/04/hWbU0.png" alt="image-20220416194816228" style="zoom:80%;" />

  过程中发生的”失真“常常会使攻击的signal丧失效应。

- **Generator**

  > https://arxiv.org/abs/1805.06605 ——目标：如何用generator重新生成的图片

  对所有输入的图片（Input Image），用generator重新生成（reconstruct）

  <img src="https://s1.328888.xyz/2022/05/04/hWFqJ.png" alt="image-20220416195041251" style="zoom:80%;" />

  利用Generator抹去加入的attack signal。

##### 局限性

- 一旦被别人（attacker）知道被动防御的措施，就立马失效。我们可以把模糊化看作是network之前多加了一层NN，那么攻击者就可以适应这种方式，重新应对。

<img src="https://s1.328888.xyz/2022/05/04/hWVmF.png" alt="image-20220416195147553" style="zoom:80%;" />

怎么办？加上自己都不知道怎么随机在哪儿的随机层（如上图所示，来自https://arxiv.org/abs/1711.01991）——欲欺敌先瞒己，乱拳打死老师傅，hhhh。

但是假设attacker知道你随机的distribution，也是有可能被攻破的。

#### Proactive Defense（主动攻击）

直接训练一个对adversarial attack具备鲁棒性的模型——

##### Adversarial Training

- 给出训练资料$X = \{(x^1,\hat{y}^1),(x^2,\hat{y}^2),...,(x^N,\hat{y}^N)\}$，使用训练资料$X$来训练模型

- 训练阶段就对模型展开攻击——

  $For \ n = 1\ to \ N$：对于每个给出的$x^n$通过攻击算法找到对应的adversarial input $\widetilde{x}^n$；这里的$\widetilde{x}^n$就是攻击成功的图片，重新进行正确的标记，从而得到新的训练资料$X' = \{(\tilde{x}^1,\hat{y}^1),(\widetilde{x}^2,\hat{y}^2),...,(\tilde{x}^N,\hat{y}^N)\}$

- 合并两个训练资料$X$和$X'$，更新模型，
- 不断重复上述过程，不断fix漏洞。用类似于Data Augmentation的方式，让这种数据驱动的模型更具备鲁棒性。

**局限性**在于不一定能挡住新的攻击方式，如果新的攻击方式不在以上这种数据增强的方式被考虑，那么防御可能无效。而且，去寻找整理$\widetilde{x}^n$的过程也非常繁复耗时。消耗运算资源。

> *达到Adversarial Training的效果，但相比下不需要额外计算消耗资源。文章：[Adversarial Training for Free!](https://arxiv.org/abs/1904.12843)

### Remarks（总结）

- 攻击：固定网络参数，训练调整攻击输入——实践证明攻击非常简单。
- Black Box Attack is possible
- 防御：被动防御 & 主动防御
- 攻击/防御手段依然在进化（evolving）

#### 攻击手段（举例来说）

• [FGSM](https://arxiv.org/abs/1412.6572)
• [Basic iterative method](https://arxiv.org/abs/1607.02533)
• [L-BFGS](https://arxiv.org/abs/1312.6199)
• [Deepfool](https://arxiv.org/abs/1511.04599)
• [JSMA](https://arxiv.org/abs/1511.07528)
• [C&W](https://arxiv.org/abs/1608.04644)
• [Elastic net attack](https://arxiv.org/abs/1709.04114)
• [Spatially Transformed](https://arxiv.org/abs/1801.02612)
• [One Pixel Attack](https://arxiv.org/abs/1710.08864)

……
# StyleCLIP: 文本驱动的图像处理
## 1. 简介

StyleGAN V2 的任务是使用风格向量进行image generation，而Clip guided Editing 则是利用CLIP (Contrastive Language-Image Pre-training ) 多模态预训练模型计算文本输入对应的风格向量变化，用文字表述来对图像进行编辑操纵风格向量进而操纵生成图像的属性。相比于Editing 模块，StyleCLIP不受预先统计的标注属性限制，可以通过语言描述自由控制图像编辑。

原论文中使用 Pixel2Style2Pixel 的 升级模型 Encode4Editing 计算要编辑的代表图像的风格向量，为尽量利用PaddleGAN提供的预训练模型本次复现中仍使用Pixel2Style2Pixel计算得到风格向量进行实验，重构效果略有下降，期待PaddleGAN跟进e4e相关工作。


## 2. 复现
StyleCLIP 模型 需要使用简介中对应提到的几个预训练模型，
本次复现使用PPGAN 提供的 在FFHQ数据集上进行预训练的StyleGAN V2 模型作为生成器，并使用Pixel2Style2Pixel模型将待编辑图像转换为对应风格向量。

CLIP模型依赖Paddle-CLIP实现。
pSp模型包含人脸检测步骤，依赖dlib框架。
除本repo外还需要安装 Paddle-CLIP 和 dlib 依赖。

整体安装方法如下。
```
pip install -e .
pip install paddleclip
pip install dlib-bin
```

[模型训练日志及权重 (提取码: ep6m)](https://pan.baidu.com/s/1IvFH-_cWf4p5BcHICfLgIA#ep6m)

### 编辑结果展示

风格向量对应的图像:
<div align="center">
    <img src="docs/imgs/stylegan2fitting-sample.png" width="300"/>
</div>

设置
> direction_offset = [ -1, 0, 1, 2, 3, 4, 5]
> beta_threshold = 0.1

从 'face' 到 'boy face' 编辑得到的图像:

<div align="center">
    <img src="docs/imgs/stylegan2clip-sample-boy.png" width="640"/>
</div>

从'face' 到 'happy face' 编辑得到的图像:

<div align="center">
    <img src="docs/imgs/stylegan2clip-sample-happy.png" width="640"/>
</div>

从'face' 到 'angry face' 编辑得到的图像:

<div align="center">
    <img src="docs/imgs/stylegan2clip-sample-angry.png" width="640"/>
</div>


从'face' 到 'face with long hair' 编辑得到的图像:

<div align="center">
    <img src="docs/imgs/stylegan2clip-sample-long-hair.png" width="640"/>
</div>


从'face' 到 'face with curl hair' (卷发) 编辑得到的图像:

<div align="center">
    <img src="docs/imgs/stylegan2clip-sample-curl-hair.png" width="640"/>
</div>

从'head with black hair'（黑发） 到 'head with gold hair'（金发）编辑得到的图像:

<div align="center">
    <img src="docs/imgs/stylegan2clip-sample-gold-hair.png" width="640"/>
</div>



## 3. 使用方法
### 训练
在StyleCLIP论文中作者研究了 3 种结合 StyleGAN 和 CLIP 的方法：
1. 文本引导的风格向量优化，使用 CLIP 模型作为损失网络对现有风格向量进行多次迭代更新，但该方法对每次处理都需要重新训练。
2. 训练 风格向量映射器，使CLIP文本特征向量映射至StyleGAN 风格向量空间，避免（1）方法的训练问题，但可控性较差，经论文对比其生成质量也不如（3）。
3. 在 StyleGAN 的 StyleSpace 中，把文本描述映射到输入图像的全局方向 (Global Direction)，进而运行自由控制图像操作强度以及分离程度，实现类似于StyleGAN Editing 模块的使用体验。

本次仅复现论文中效果最好的 （3）Global Direction 方法。

StyleCLIP Global Direction 训练过程分两步：
1. 提取风格向量并统计

```
python styleclip_getf.py
```
2. 结合CLIP模型计算转换矩阵

```
python ppgan/apps/styleganv2clip_predictor.py extract
```

### 编辑
详见 [demo](./demo.ipynb)

用户使用如下命令中对图像属性进行编辑：

```
cd applications/
python -u tools/styleganv2clip.py \
       --latent <替换为要编辑的风格向量的路径> \
       --output_path <替换为生成图片存放的文件夹> \
       --weight_path <替换为你的预训练模型路径> \
       --model_type ffhq-config-f \
       --size 1024 \
       --style_dim 512 \
       --n_mlp 8 \
       --channel_multiplier 2 \
       --direction_path <替换为存放统计数据的文件路径> \
       --neutral <替换为对原图像的描述，如face> \
       --target <替换为对目标图像的描述> \
       --beta_threshold 0.12 \
       --direction_offset 5
       --cpu
```

**参数说明:**
- latent: 要编辑的代表图像的风格向量的路径。可来自于Pixel2Style2Pixel生成的`dst.npy`或StyleGANv2 Fitting模块生成的`dst.fitting.npy`
- output_path: 生成图片存放的文件夹
- weight_path: 或StyleGANv2 预训练模型路径
- model_type: 模型类型,当前使用: `ffhq-config-f`
- direction_path: 存放CLIP统计向量的文件的路径。默认为空，即使用'fs3.npy'。若不使用，请在命令中去除
- neutral: 对原图像的中性描述，如 face
- target: 为对目标图像的描述，如 young face
- beta_threshold: 向量调整阈值
- direction_offset: 属性的偏移强度
- cpu: 是否使用cpu推理，若不使用，请在命令中去除

!以下 参数需与StyleGAN 预训练模型保持一致
- size: 模型参数，输出图片的分辨率
- style_dim: 模型参数，风格z的维度
- n_mlp: 模型参数，风格z所输入的多层感知层的层数
- channel_multiplier: 模型参数，通道乘积，影响模型大小和生成图片质量

## 复现记录
1. PaddleGAN 实现中的StyleGAN模型将Style Affine层进行了模块耦合，而论文中使用到的S Space 需要用到，因此对StyleGAN 生成器代码也进行了魔改，增加style_affine 及 synthesis_from_styles 方法同时兼容现有接口。
2. Paddle Resize处理对Tensor和ndarray的处理方法不同，默认Tensor使用BCHW模式存储而非图像的BHWC。
3. 现有 uppfirdn2d 模块中似乎存在多次不必要的Tensor拷贝、reshape过程，希望后续能够优化运算及显存占用。
4. 切片拷贝：paddle中对Tensor进行切片时（有时）会创建新的拷贝，此时再对其进行赋值很可能不生效，两种写法`a[ind1][ind2]=0` 和 `a[ind1, ind2]=0` 前者并不改变a中的参数。
5. StyleCLIP论文中表示使用100张图像进行Global Direction 训练在V1080Ti需要约4h,但使用V100的训练数据及官方repo中也有issue提到实际需要约24h，该问题但作者还未能给出解答。

## 参考repo
- [StyleCLIP](https://github.com/orpatashnik/StyleCLIP)
- [StyleCLIP-pytorch](https://github.com/soushirou/StyleCLIP-pytorch)
- [PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN)
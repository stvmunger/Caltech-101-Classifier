# Caltech-101 Classifier

> 基于 **PyTorch 2.7 + CUDA 12.8** 的 ResNet-18 图像分类示例，  
> 演示 *scratch* 训练与 *ImageNet fine-tune* 在 **Caltech-101** 数据集上的性能差异。  
>  Fine-tune 30 epoch 可达 **95.9 %** Top-1 Accuracy。

---

# 1  目录结构

Caltech-101/
│

├─ data/ # 原始数据

 #│ └─ 101_ObjectCategories/

├─ scripts/     
        #│ └─ split_dataset.py # 数据划分 30/剩余

├─ models/ # 保存 .pth 权重 (已 .gitignore)

├─ logs/ # TensorBoard 日志

├─ config.py # 超参数 & 路径

├─ utils.py # dataloader / metrics

├─ model.py # ResNet-18/AlexNet 构建

├─ train.py # 训练主循环

├─ test.py # 加载 checkpoint 做评估

├─ main.py 

└─ README.md

# 2  环境配置

Ubuntu 22.04, CUDA 12.8 驱动 570.xx, PyTorch 2.7.0+cu128

1) 建议 Conda
conda create -n cal101 python=3.10 -y
conda activate cal101

1) GPU 版 PyTorch 2.7 (CUDA 12.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

1) 其它依赖
pip install scikit-learn tensorboard pandas tqdm

# 3  数据准备
#官方 zip ⇒ 解压后得到 101_ObjectCategories 目录

mkdir -p data

mv 101_ObjectCategories data/

# 4  训练 & 测试
## 4.1 Scratch（随机初始化）
python main.py \
  --mode train \
  --backbone resnet18 \
  --epochs 30 \

## 4.2 Fine-tune（加载 ImageNet 权重）
python main.py \
  --mode train \
  --backbone resnet18 \
  --pretrained \
  --epochs 30

#日志默认写入 logs/resnet18_{scratch|pre}；模型权重保存在 models/，文件名形如 resnet18_pre_e30.pth。

## 4.3 评估

#scratch

python main.py --mode test --backbone resnet18 --epochs 30

#fine-tune

python main.py --mode test --backbone resnet18 --pretrained --epochs 30

# 5  TensorBoard 可视化

tensorboard --logdir logs --host 0.0.0.0 --port 6006

#在浏览器访问 http://<WSL_IP>:6006/，查看 train/loss 与 val/accuracy 曲线。

# 6 结果
模型	Top-1 Acc.	Macro F1	备注

ResNet-18 scratch	52 %	29 %	30 epoch

ResNet-18 fine-tune	95.9 %	95.0 %	30 epoch
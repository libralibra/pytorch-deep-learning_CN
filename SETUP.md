# 编写 PyTorch 代码的设置

设置用于编写深度学习代码的机器可能有些麻烦。

涵盖了从硬件到软件再到所有需要注意的小问题，为了让代码像在您的机器上运行一样在其他人的机器上运行。

为了本课程，我们尽量保持简单。

尽管不是那么简单，但是您将无法在其他地方使用这里一样的设置。

有两个设置选项。一个容易些，但另一个则提供了更多的选择。
1. 使用Google Colab（最简单）
2. 设置您自己的本地/远程计算机（需要额外几个步骤，但有更多的灵活性）

**注意：** 这些都不是为了替代 [Pytorch的官方设置文档](https://pytorch.org/get-started/locally/), 如果您想长期使用Pytorch编写代码，则应该熟悉它。

## 1. Google Colab的设置（最简单）

Google Colab是一种免费的在线交互式计算引擎（基于Jupyter笔记本电脑，数据科学标准）。

Google Colab的好处是：
* 几乎为零的设置（Google Colab配备了Pytorch和许多其他数据科学软件包，例如Pandas，Numpy和Matplotlib已安装）
* 使用链接分享您的工作
* 免费访问GPU（gpus使您的深度学习代码更快），带有付费的选项，可以访问 *更多* GPU Power

Google Colab的缺点是：
* 超时（大多数COLAB笔记本仅保留最高2-3个小时的状态，尽管这可以随着付费选项而增加）
* 无法访问本地存储（尽管有解决方案）
* 脚本的设置不是很好（将代码转换为模块）

### 使用Google Colab开始，在需要时进行扩展

对于课程的入门笔记本（00-04），我们将仅使用Google Colab。

这是因为它不仅满足我们的需求。

实际上，这是我经常自己做的工作流程。

我在Google Colab中做了大量的初学者和实验性工作。

而且，当我找到想变成一个更大的项目或更多工作的东西时，我会移至本地计算或云托管计算。

### Google Colab入门

要从Google Colab开始，我首先要进行[Google colab笔记本简介]（https://colab.research.google.com/notebooks/notebooks/basic_features_overview.ipynb）（只是为了熟悉所有的设置与操作）。

### 一键打开课程笔记本

After you've gotten familiar with the Google Colab interface, you can run any of the course notebooks directly in Google Colab by pressing the "Open in Colab" button at the top of the online book version or the GitHub version.

![open a course notebook in Google Colab via open in Colab button](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/setup-open-in-colab-cropped.gif)

If you'd like to make a copy of the notebook and store it on your Google Drive, you can press the "Copy to Drive" button.

### Opening a notebook in Google Colab with a link

You can also enter any notebook link from GitHub directly in Google Colab and get the same result.

![open a course notebook in Google Colab via GitHub link](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/setup-open-notebook-in-colab-via-link.png)

Doing this will give you a runable notebook right in Google Colab. 

Though this should only be used for testing purposes, as when going through the course, I highly recommend you **write the code yourself**, rather than running existing code.

### Getting acess to a GPU in Google Colab

To get access to a CUDA-enabled NVIDIA GPU (CUDA is the programming interface that allows deep learning code to run faster on GPUs) in Google Colab you can go to `Runtime -> Change runtime type -> Hardware Accelerator -> GPU` (note: this will require the runtime to restart).

![Getting access to a GPU in Google Colab](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/setup-get-gpu-colab-cropped.gif)

To check whether you have a GPU active in Google Colab you can run:

```
!nvidia-smi
```

If you have access to a GPU, this will show you what kind of GPU you have access to.

And to see if PyTorch has access to the GPU, you can run:

```python
import torch # Google Colab comes with torch already installed
print(torch.cuda.is_available()) # will return True if PyTorch can use the GPU
```

If PyTorch can see the GPU on Google Colab, the above will print `True`.

## TK - 2. Getting setup locally (Linux version)

> **Note:** A reminder this is not a replacement for the [PyTorch documentation for getting setup locally](https://pytorch.org/get-started/locally/). This is only one way of getting setup (there are many) and designed specifically for this course.

This **setup is focused on Linux systems** (the most common operating system in the world), if you are running Windows or macOS, you should refer to the PyTorch documentation. 

This setup also **expects you to have access to a NVIDIA GPU**.

Why this setup?

As a machine learning engineer, I use it almost daily. It works for a large amount of workflows and it's flexible enough so you can change if you need.

Let's begin.

### Setup steps locally for a Linux system with a GPU
TK TODO - add step for install CUDA drivers
TK image - overall setup of the course environment (e.g. Jupyter Lab inside conda env)

1. [Install Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) (you can use Anaconda if you already have it), the main thing is you need access to `conda` on the command line. Make sure to follow all the steps in the Miniconda installation guide before moving onto the next step.
2. Make a directory for the course materials, you can name it what you want and then change into it. For example:
```
mkdir ztm-pytorch-course
cd ztm-pytorch-course
```
3. Create a `conda` environment in the directory you just created. The following command will create a `conda` enviroment that lives in the folder called `env` which lives in the folder you just created (e.g. `ztm-pytorch-course/env`). Press `y` when the command below asks `y/n?`.
```
conda create --prefix ./env python=3.8.13
```
4. Activate the environment you just created.
```
conda activate ./env
```
5. Install the code dependencies you'll need for the course such as PyTorch and CUDA Toolkit for running PyTorch on your GPU. You can run all of these at the same time (**note:** this is specifically for Linux systems with a NVIDIA GPU, for other options see the [PyTorch setup documentation](https://pytorch.org/get-started/locally/)):
```
conda install -c pytorch pytorch=1.10.0 torchvision cudatoolkit=11.3 -y
conda install -c conda-forge jupyterlab torchinfo torchmetrics -y
conda install -c anaconda pip -y
conda install pandas matplotlib scikit-learn -y
```
6. Verify the installation ran correctly by running starting a Jupyter Lab server:

```bash
jupyter lab
```

7. After Jupyter Lab is running, start a Jupyter Notebook and running the following piece of code in a cell.
```python
import pandas as pd
import numpy as np
import torch
import sklearn
import matplotlib
import torchinfo, torchmetrics

# Check PyTorch access (should print out a tensor)
print(torch.randn(3, 3))

# Check for GPU (should return True)
print(torch.cuda.is_available())
```

If the above code runs without errors, you should be ready to go.

If you do run into an error, please refer to the [Learn PyTorch GitHub Discussions page](https://github.com/mrdbourke/pytorch-deep-learning/discussions) and ask a question or the [PyTorch setup documentation page](https://pytorch.org/get-started/locally/).
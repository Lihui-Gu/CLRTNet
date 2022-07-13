<!--
 * @Author: 顾立辉 glh9803@outlook.com
 * @Date: 2022-07-12 23:02:30
 * @LastEditors: 顾立辉 glh9803@outlook.com
 * @LastEditTime: 2022-07-13 22:35:34
 * @FilePath: \undefinedc:\Users\glh98\Documents\GitHub\CLRTNet\README.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
## Introduction
I investigated and summerized the curent mainstream lane detection model, improved and adjusted the model based on [CLRNet](https://github.com/Turoad/CLRNet), built a SpaceFormer with spatial attention mechanism to obtain context information, completed the construction of a new network structure CLRTNet, and test it on tusimple Dataset. The final result is very close to the accuracy level of CLRNet. 
## Installation

### Prerequisites
Only test on Ubuntu18.04 and 20.04 with:
- Python >= 3.8 (tested with Python3.8)
- PyTorch >= 1.6 (tested with Pytorch1.6)
- CUDA (tested with cuda10.2)
- Other dependencies described in `requirements.txt`

### Clone this repository
Clone this code to your workspace. 
We call this directory as `$CLRNET_ROOT`
```Shell
git clone https://github.com/Turoad/clrnet
```

### Create a conda virtual environment and activate it (conda is optional)

```Shell
conda create -n clrnet python=3.8 -y
conda activate clrnet
```

### Install dependencies

```Shell
# Install pytorch firstly, the cudatoolkit version should be same in your system.

conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

# Or you can install via pip
pip install torch==1.8.0 torchvision==0.9.0

# Install python packages
python setup.py build develop
```

### Data preparation Tusimple
Download [Tusimple](https://github.com/TuSimple/tusimple-benchmark/issues/3). Then extract them to `$TUSIMPLEROOT`. Create link to `data` directory.

```Shell
cd $CLRNET_ROOT
mkdir -p data
ln -s $TUSIMPLEROOT data/tusimple
```

For Tusimple, you should have structure like this:
```
$TUSIMPLEROOT/clips # data folders
$TUSIMPLEROOT/lable_data_xxxx.json # label json file x4
$TUSIMPLEROOT/test_tasks_0627.json # test tasks json file
$TUSIMPLEROOT/test_label.json # test label json file

```

For Tusimple, the segmentation annotation is not provided, hence we need to generate segmentation from the json annotation. 

```Shell
python tools/generate_seg_tusimple.py --root $TUSIMPLEROOT
# this will generate seg_label directory
```
## Getting Started

### Training
For training, run
```Shell
python main.py [configs/path_to_your_config] --gpus [gpu_num]
```

For example, run
```Shell
python main.py configs/clrnet/clr_resnet18_culane.py --gpus 0
```

### Validation
For testing, run
```Shell
python main.py [configs/path_to_your_config] --[test|validate] --load_from [path_to_your_model] --gpus [gpu_num]
```

For example, run
```Shell
python main.py configs/clrnet/clr_dla34_culane.py --validate --load_from culane_dla34.pth --gpus 0
```

Currently, this code can output the visualization result when testing, just add `--view`.
We will get the visualization result in `work_dirs/xxx/xxx/visualization`.

## Loss Optimization strategy
Use dynamic weight adjust strategy, see more at `global_var.py`.


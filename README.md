SALMNet: A structure-aware lane marking detection network
======
by Xuemiao Xu, Tianfei Yu, Xiaowei Hu, Wing W. Y. Ng, and Pheng-Ann Heng[paper link](https://ieeexplore.ieee.org/abstract/document/9061152)<br>
This implementation is written by Tianfei Yu<br>

Citation
-----
@article{xu2020salmnet,<br>
  title={SALMNet: A structure-aware lane marking detection network},<br>
  author={Xu, Xuemiao and Yu, Tianfei and Hu, Xiaowei and Ng, Wing WY and Heng, Pheng-Ann},<br>
  journal={IEEE Transactions on Intelligent Transportation Systems},<br>
  year={2020},<br>
  publisher={IEEE}<br>
}<br>

Requirement
-----
Python 2.7<br>
Pytorch 0.4.1<br>
torchvision<br>
numpy<br>

Training
-----
Set the path of pretrained ResNet model in config<br>
Set the path of datasets in datasets<br>
Run by `python train.py`<br>

Testing
-----
Run by `python eval.py`<br>

Relevant Links
-----
[Deformable convolution implementation](https://github.com/1zb/deformable-convolution-pytorch)
[Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch)


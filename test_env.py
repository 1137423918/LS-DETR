
# mmcv、mmengin测试代码
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d, modulated_deform_conv2d
from mmcv.cnn import ConvModule, build_norm_layer
from mmengine.model import BaseModule
from mmengine.model import constant_init
from mmengine.model.weight_init import trunc_normal_init, normal_init

# mamba测试代码
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

# dcnv3测试代码
import DCNv3
import pkg_resources
dcn_version = float(pkg_resources.get_distribution('DCNv3').version)

# dcnv4测试代码
from DCNv4.modules.dcnv4 import DCNv4

# smpconv测试代码
from depthwise_conv2d_implicit_gemm import _DepthWiseConv2dImplicitGEMMFP16, _DepthWiseConv2dImplicitGEMMFP32

# mamba-yolo测试代码
import selective_scan_cuda_core
from typing import Optional, Union
import torch
from torch import nn
import numpy as np
import random
import torch.nn.functional as F
from sklearn.decomposition import PCA

glo_count = 0
alpha = 0

class SVDLoRALinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_1_a: torch.Tensor,
        weight_1_b: torch.Tensor,
        weight_2_a: torch.Tensor,
        weight_2_b: torch.Tensor,
        interp_kv: bool = False,
        method: str = 'direct_linear',
        rank: int = 8,
        device: Optional[Union[torch.device, str]] = "cuda",
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.device = device
        self.weight_1_a = weight_1_a.to(device)
        self.weight_1_b = weight_1_b.to(device)
        self.weight_2_a = weight_2_a.to(device)
        self.weight_2_b = weight_2_b.to(device)
        self.interp_kv = interp_kv
        self.method = method
        self.cached_weights = None

        self.linear_layer1 = nn.Linear(in_features=rank, out_features=rank).to(
            self.device
        )
        self.linear_layer2 = nn.Linear(in_features=rank, out_features=rank).to(
            self.device
        )
        self.linear_layer3 = nn.Linear(in_features=rank, out_features=rank).to(
            self.device
        )
        self.linear_layer4 = nn.Linear(in_features=rank, out_features=rank).to(
            self.device
        )

        self.out_features = out_features
        self.in_features = in_features
        self.forward_type = "merge"

    def set_forward_type(self, type: str = "merge"):
        assert type in ["merge", "weight_1", "weight_2"]
        self.forward_type = type

    def get_svdmix_weight(self):
        # 保存原始数据类型
        orig_dtype = self.weight_1_a.dtype

        # 转 float32 以支持 SVD
        W1 = self.weight_1_a.to(torch.float32).to(self.device)
        W2 = self.weight_2_a.to(torch.float32).to(self.device)
        W1_ = self.weight_1_b.to(torch.float32).to(self.device)
        W2_ = self.weight_2_b.to(torch.float32).to(self.device)

        # 第一次融合：如果有缓存就直接使用
        if self.cached_weights is not None:
            U1, S1, V1, U2, S2, V2, U1_, S1_, V1_, U2_, S2_, V2_ = self.cached_weights
        else:
            # 对每个矩阵进行 SVD 分解
            U1, S1, V1 = torch.linalg.svd(W1, full_matrices=False)
            U2, S2, V2 = torch.linalg.svd(W2, full_matrices=False)
            U1_, S1_, V1_ = torch.linalg.svd(W1_, full_matrices=False)
            U2_, S2_, V2_ = torch.linalg.svd(W2_, full_matrices=False)

            self.cached_weights = (U1, S1, V1, U2, S2, V2,
                                U1_, S1_, V1_, U2_, S2_, V2_)

        # 在子空间中进行均值融合
        U_mean = alpha * U1 + (1 - alpha) * U2
        S_mean = alpha * S1 + (1 - alpha) * S2
        V_mean = alpha * V1 + (1 - alpha) * V2

        U_mean_ = alpha * U1_ + (1 - alpha) * U2_
        S_mean_ = alpha * S1_ + (1 - alpha) * S2_
        V_mean_ = alpha * V1_ + (1 - alpha) * V2_

        # 重构融合后的矩阵
        Sigma_mean = torch.diag(S_mean)
        Sigma_mean_ = torch.diag(S_mean_)
        W_fusion1 = U_mean @ Sigma_mean @ V_mean
        W_fusion2 = U_mean_ @ Sigma_mean_ @ V_mean_

        # 合并结果
        result = W_fusion2 @ W_fusion1

        # 转回原始类型（例如 bfloat16）
        return result.to(orig_dtype)

    def get_linear_weight(self, timestep):
        global alpha
        # matrix1对应content matrix2对应style
        matrix1 = self.weight_1_b @ self.weight_1_a
        matrix2 = self.weight_2_b @ self.weight_2_a
        # print(alpha)
        return alpha * matrix1 + (1.0 - alpha) * matrix2


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        global glo_count  
        orig_dtype = hidden_states.dtype
        dtype = self.weight_1_a.dtype

        if self.forward_type == "merge":
            glo_count += 1
            method_func_map = {

                "svdmix": self.get_svdmix_weight,

                "direct_linear": lambda: self.get_linear_weight(glo_count),
            }
            weight = method_func_map[self.method]()
        elif self.forward_type == "weight_1":
            weight = self.weight_1_b @ self.weight_1_a
        elif self.forward_type == "weight_2":
            weight = self.weight_2_b @ self.weight_2_a
        else:
            raise ValueError(self.forward_type)
        hidden_states = nn.functional.linear(
            hidden_states.to(dtype), weight
        )

        return hidden_states.to(orig_dtype)


class SVDLoRALinearLayerInference(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.zeros((out_features, in_features),
                        device=device, dtype=dtype),
            requires_grad=False,
        )
        self.out_features = out_features
        self.in_features = in_features

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.weight.dtype
        hidden_states = nn.functional.linear(
            hidden_states.to(dtype), weight=self.weight
        )
        return hidden_states.to(orig_dtype)

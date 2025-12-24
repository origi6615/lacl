import torch
from torch.autograd import Variable
import numpy as np
from .ACIL import ACILLearner
from typing import Callable, Dict, Any, Optional, Sequence
from .AnalyticLinear import AnalyticLinear, RecursiveLinear
from .Buffer import activation_t, RandomBuffer
from torch._prims_common import DeviceLikeType
from torch.nn import functional as F
from sklearn.neighbors import NearestNeighbors


class LACL(torch.nn.Module):
    def __init__(
        self,
        backbone_output: int,
        backbone: Callable[[torch.Tensor], torch.Tensor] = torch.nn.Flatten(),
        expansion_size: int = 8192,
        gamma_main: float = 1e-3,
        gamma_comp: float = 1e-3,
        C: float = 1,
        k: int = 500,
        metric: str = "cosine",  
        device=None,
        dtype=torch.double,
        linear: type[AnalyticLinear] = RecursiveLinear,
    ) -> None:
        super().__init__()

        if metric not in ["cosine", "euclidean"]:
            raise ValueError(f"use 'cosine'or 'euclidean'")
            
        factory_kwargs = {"device": device, "dtype": dtype}
        self.backbone = backbone
        self.expansion_size = expansion_size
        self.buffer = RandomBuffer(
            backbone_output,
            expansion_size,
            activation=torch.nn.Identity(),** factory_kwargs
        )

        self.activation_main = torch.relu  
        self.main_stream = linear(expansion_size, gamma_main, **factory_kwargs)

        self.C = C
        self.k = k  
        self.metric = metric  
        self.comp_stream = linear(expansion_size, gamma_comp,** factory_kwargs)
        self.eval()

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.buffer(self.backbone(X))
        X_main = self.main_stream(self.activation_main(X))

        X_comp = self.comp_stream(torch.tanh(self._knn_mean(X)))
        return X_main + self.C * X_comp

    def _knn_mean(self, X: torch.Tensor) -> torch.Tensor:
        X_np = X.cpu().numpy()
        num_samples = X_np.shape[0]
        

        k_eff = min(self.k, num_samples - 1)
        if k_eff < 1:

            return X
    

        knn = NearestNeighbors(n_neighbors=k_eff + 1, metric=self.metric)
        knn.fit(X_np)
        _, indices = knn.kneighbors(X_np)
        indices = indices[:, 1:]  
    
        X_knn = []
        for i in range(num_samples):
            neighbor_features = X_np[indices[i]]
            mean_feature = np.mean(neighbor_features, axis=0)
            X_knn.append(mean_feature)
    
        return torch.tensor(np.array(X_knn), device=X.device, dtype=X.dtype)

    @torch.no_grad()
    def fit(self, X: torch.Tensor, y: torch.Tensor, increase_size: int) -> None:
        num_classes = max(self.main_stream.out_features, int(y.max().item()) + 1)
        Y_main = torch.nn.functional.one_hot(y, num_classes=num_classes)
        X = self.buffer(self.backbone(X))


        X_main = self.activation_main(X)
        self.main_stream.fit(X_main, Y_main)
        self.main_stream.update()


        Y_comp = Y_main - self.main_stream(X_main)
        Y_comp[:, :-increase_size] = 0


        X_comp = torch.tanh(self._knn_mean(X))
        self.comp_stream.fit(X_comp, torch.tanh(Y_comp))

    @torch.no_grad()
    def update(self) -> None:
        self.main_stream.update()
        self.comp_stream.update()


class LACLLearner(ACILLearner):

    def __init__(
        self,
        args: Dict[str, Any],
        backbone: torch.nn.Module,
        backbone_output: int,
        device=None,
        all_devices: Optional[Sequence[DeviceLikeType]] = None,
    ) -> None:
        self.gamma_comp = args["gamma_comp"]
        self.compensation_ratio = args["compensation_ratio"]
        self.k = args.get("k", 5)  

        self.metric = args.get("metric", "cosine")
        super().__init__(args, backbone, backbone_output, device, all_devices)

    def make_model(self) -> None:
        self.model = LACL(
            self.backbone_output,
            self.backbone,
            self.buffer_size,
            self.gamma,
            self.gamma_comp,
            self.compensation_ratio,
            k=self.k,
            metric=self.metric,  
            device=self.device,
            dtype=torch.double,
            linear=RecursiveLinear,
        )

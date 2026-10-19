from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from patch_qc.model import DinoBinaryClassifier
from patch_qc.train import train_one_epoch


class TinyPatchBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, 8, kernel_size=16, stride=16, bias=False)

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        patches = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return SimpleNamespace(last_hidden_state=torch.cat((patches.mean(1, keepdim=True), patches), 1))


class TinyDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "image": torch.full((3, 256, 256), index / 4, dtype=torch.float32),
            "label": torch.tensor(float(index % 2), dtype=torch.float32),
        }


def test_train_one_epoch_updates_head_without_backbone_gradients() -> None:
    torch.manual_seed(3)
    model = DinoBinaryClassifier(TinyPatchBackbone(), hidden_size=8, dropout=0.0)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3)
    loader = DataLoader(TinyDataset(), batch_size=2, shuffle=False)

    metrics = train_one_epoch(model, loader, optimizer, device=torch.device("cpu"))

    assert metrics["examples"] == 4
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["loss"] > 0.0
    assert all(parameter.grad is None for parameter in model.backbone.parameters())

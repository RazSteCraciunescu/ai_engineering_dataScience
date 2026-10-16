"""Optional PyTorch MLP training loop with explicit tensors and gradients."""
from __future__ import annotations


def main() -> None:
    import torch
    from torch import nn

    torch.manual_seed(42)
    x = torch.randn(320, 4)
    y = ((x[:, 0] + 0.8 * x[:, 1] - 0.3 * x[:, 2]) > 0).long()
    model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(120):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

    accuracy = (model(x).argmax(dim=1) == y).float().mean()
    print({"loss": float(loss.detach()), "accuracy": float(accuracy)})


if __name__ == "__main__":
    main()

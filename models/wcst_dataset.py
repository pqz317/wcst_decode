import torch
from torch.utils.data import Dataset


class WcstDataset(Dataset):
    """
    PyTorch Dataset for WCST decoder training
    """

    def __init__(self, x, y, cards=None):
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.x = torch.tensor(x).float().to(device)
        self.y = torch.tensor(y).to(device)
        self.cards = None
        if cards is not None:
            self.cards = torch.tensor(cards).to(torch.long).to(device)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.cards is not None:
            return self.x[idx], self.y[idx], self.cards[idx]
        else:
            return self.x[idx], self.y[idx], None


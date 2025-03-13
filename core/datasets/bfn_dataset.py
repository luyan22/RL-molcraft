from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch

class BFNDataset(Dataset):
    def __init__(self, ref_model, num_samples=1e5):
        self.ref_model = ref_model
        self._length = int(num_samples)  # 虚拟长度，控制epoch大小

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        # 动态生成数据（每次调用独立采样）
        sample = self.ref_model.sampling() 
        # 假设返回格式为字典，需转换为张量
        return {
            'input': torch.tensor(sample['input']), 
            'target': torch.tensor(sample['target'])
        }

class DataModule(pl.LightningDataModule):
    def __init__(self, ref_model, batch_size=32):
        super().__init__()
        self.ref_model = ref_model
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.dataset = BFNDataset(self.ref_model)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=4,  # 注意多进程问题
            persistent_workers=True
        )

# Guide3D

## TODO's

- [x] Clean and document the segmentation code
- [ ] Clean and document the image based code
- [ ] Clean the repository
- [ ] Add quick-start
- [ ] Add detailed explanation on README
- [ ] Explain the folder structure

## Quickstart

```python
from torch.utils import data

from guide3d.dataset.image.segment import Guide3D

dataset = Guide3D(
    dataset_path="~/datasets/test",
    annotations_file="raw.json",
    split="train",
    split_ratio=(0.8, 0.1, 0.1),
    download=True,
)

dataloader = data.DataLoader(dataset, batch_size=1)

batch = next(iter(dataloader))
for batch in dataloader:
    img, mask = batch
    print(img.shape)
    print(mask.shape)
    exit()
```

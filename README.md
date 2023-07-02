# PyTorch MeZO optimizer

Unofficial minimal implementation of MeZO optimizer.  
For research purposes.  

> [<b>Fine-Tuning Language Models with Just Forward Passes</b>]([https://link-url-here.org](https://arxiv.org/abs/2305.17333)https://arxiv.org/abs/2305.17333).   
Official repo [(here)](https://github.com/princeton-nlp/MeZO)

## How to use
Simply copy paste `mezo.py` in your repository and import the optimizer.

```python
from mezo import MeZO

opt = MeZO(torch.optim.SGD(model.parameters(), lr=0.05), eps=1e-3) 
opt = MeZO(torch.optim.AdamW(model.parameters(), lr=0.005), eps=1e-3)  
```

## Disclaimer
Work in progress. May have bugs. Use at your discretion.

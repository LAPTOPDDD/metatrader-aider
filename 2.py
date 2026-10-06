import torch
print(torch.__version__)
print(torch.cuda.is_available())  # должно быть False
from tabpfn import TabPFNRegressor
print("TabPFN импортирован успешно")
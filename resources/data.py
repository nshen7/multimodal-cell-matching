import anndata as ad
import numpy as np
from torch.utils.data import Dataset, DataLoader

def get_chr_index(adata_atac):
  r"""
  Output row indices for each chromosome for each chromosome
  Parameters
  ----------
  adata_atac
      annData for ATAC
  Returns
  -------
  chr_index
      Dictionary of indices for each chromosome
  """
  row_name = adata_atac.var.index
  chr_name = [c.split("-")[0] for c in row_name]
  lst = np.unique(chr_name) # names for chromosome

  chr_index = dict()
  for i in range(len(lst)):
    index = [a for a, l in enumerate(chr_name) if l == lst[i]]
    if lst[i] not in chr_index:
      chr_index[lst[i]]=index

  return chr_index


class MultiomeDataset(Dataset):
    def __init__(
        self, csr_gex, csr_atac, cell_type
    ):
        super().__init__()
        
        self.csr_gex = csr_gex
        self.csr_atac = csr_atac
        self.cell_type = cell_type
    
    def __len__(self):
        return self.csr_gex.shape[0]
    
    def __getitem__(self, index: int):
        x_gex = torch.tensor(self.csr_gex[index,:].todense())
        x_atac = torch.tensor(self.csr_atac[index,:].todense())
        return {'gex':x_gex, 'atac':x_atac, 'cell_type':self.cell_type[index]}
  
def get_dataloaders(gex_train, atac_train, cell_type_train,  gex_val, atac_val, cell_type_val):
    
    # mod2_train = mod2_train.iloc[sol_train.values.argmax(1)]
    # mod2_test = mod2_test.iloc[sol_test.values.argmax(1)]
    
    dataset_train = MultiomeDataset(gex_train, atac_train, cell_type_train)
    data_train = DataLoader(dataset_train, config.BATCH_SIZE, shuffle = True, num_workers = config.NUM_WORKERS)
    
    dataset_val = MultiomeDataset(gex_val, atac_val, cell_type_val)
    data_val = DataLoader(dataset_val, config.BATCH_SIZE, shuffle = False, num_workers = config.NUM_WORKERS)
    
    return data_train, data_val
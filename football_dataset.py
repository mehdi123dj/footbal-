    
import os
from typing import Callable, List, Optional
import torch
from torch_geometric.data import Data,InMemoryDataset, extract_zip
from data_reader import Match_attr
import random 

class MyFootBallDataset(InMemoryDataset):
    def __init__(
                self,
                root: str, 
                split = 'train',
                transform: Optional[Callable] = None,
                pre_transform: Optional[Callable] = None,
                ):
        super().__init__(root, transform, pre_transform)

        if split == 'train':
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif split == 'val':
            self.data, self.slices = torch.load(self.processed_paths[1])
        elif split == 'test':
            self.data, self.slices = torch.load(self.processed_paths[2])

    @property
    def raw_file_names(self) -> List[str]:
        return ['database.sqlite']
    
    @property
    def processed_file_names(self) -> str:
        return ['train.pt', 'val.pt', 'test.pt']

    def process(self):

        Matchs = Match_attr(os.path.join(self.root,'raw',self.raw_file_names[0]))
        DATA = Matchs.compute_graphs()
        retained = [i for i in range(len(DATA)) if set(DATA[i].x.flatten().tolist()) != {0.0} and set(DATA[i].edge_attr.flatten().tolist()) != {0.0} ]        
        DATA = [DATA[i] for i in retained]

        train_idxs = random.choices([i for i in range(len(DATA))], k=int(len(DATA)*.8))
        val_test_idxs = list(set([i for i in range(len(DATA))])-set(train_idxs))
        val_idxs = random.choices(val_test_idxs, k=int(len(val_test_idxs)*.5))
        test_idxs = list(set(val_test_idxs) - set(val_idxs))

        #config = {'train' : (0,int(len(DATA)*.8)),
        #          'val'   : (int(len(DATA)*.8),int(len(DATA)*.9)),
        #          'test'  : (int(len(DATA)*.9),len(DATA))}

        config = {'train' : train_idxs,
                  'val'   : val_idxs,
                  'test'  : test_idxs}

        for i,s in enumerate(list(config.keys())):

            #data_list = DATA[config[s][0]:config[s][1]]
            data_list = [DATA[u] for u in  config[s]]

            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[i])

    @property
    def num_classes(self) -> int:
        return 3

MyFootBallDataset("/home/mehdi/Desktop/Datarvest/Projects/Demo/football/data")


from torch_geometric.nn.conv import GATConv, TransformerConv 
from torch_geometric.nn import Linear
from torch.nn import Softmax, CrossEntropyLoss 
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from torch.nn.functional import one_hot
import torch 
import warnings
warnings.filterwarnings('always') 


class Transformer_model(torch.nn.Module):
    def __init__(self,in_channels_node,in_channels_edges,hidden, out_channels, heads,num_classes,dropout):
        super().__init__()

        #First encode nodes and edges features with multi-attention head process
        self.conv1 = TransformerConv(   in_channels = in_channels_node,
                                        out_channels = hidden, 
                                        heads = 1,
                                        dropout = dropout,
                                        edge_dim = in_channels_edges
                                                            )
        self.conv2 = TransformerConv(   in_channels = -1,
                                        out_channels = out_channels, 
                                        heads = 1,
                                        dropout = dropout,
                                        edge_dim = in_channels_edges
                                        )
        self.lin = Linear(out_channels,num_classes)


    def forward(self, data):
        H1 = self.conv1(data.x.float(), data.edge_index.long(),data.edge_attr.float())
        H2 = self.conv2(H1, data.edge_index.long(),data.edge_attr.float())
        output = self.lin(H2)
        softmax = Softmax(dim=1)
        preds = softmax(output)
        return preds

class GAT_model(torch.nn.Module):
    def __init__(self,in_channels_node,in_channels_edges,hidden, out_channels, heads,num_classes,dropout):
        super().__init__()

        #First encode nodes and edges features with multi-attention head process
        self.conv1 = GATConv(   in_channels = in_channels_node,
                                out_channels = hidden, 
                                heads = 1,
                                dropout = dropout,
                                edge_dim = in_channels_edges,
                                add_self_loops = False
                            )
        self.conv2 = GATConv(   in_channels = -1,
                                out_channels = out_channels, 
                                heads = 1,
                                dropout = dropout,
                                edge_dim = in_channels_edges,
                                add_self_loops = False
                            )

        self.lin = Linear(out_channels,num_classes)

        
    def forward(self, data):
        H1 = self.conv1(data.x.float(), data.edge_index.long(),data.edge_attr.float())
        H2 = self.conv2(H1, data.edge_index.long(),data.edge_attr.float())
        output = self.lin(H2)
        softmax = Softmax(dim=1)
        preds = softmax(output)
        return preds


def train(model,train_loader,optimizer,device):

    loss_op = CrossEntropyLoss()
    total_loss = 0
    for data in train_loader : 
        model.train()
        optimizer.zero_grad()
        data = data.to(device)
        x_out  = model(data)

        loss = loss_op(x_out, data.y.argmax(dim=1))
        total_loss += loss.item() / len(train_loader.dataset)
        loss.backward()
        optimizer.step()
    return total_loss 


def test(model,loader,device):

    ys, preds = [], []
    for data in loader:
        model.eval()
        data = data.to(device)
        x_out = model(data)
        preds.append(torch.nn.functional.one_hot(torch.tensor([x_out[i].argmax().item() for i in range(x_out.shape[0])]),3))
        ys.append(data.y)

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()

    roc_auc = roc_auc_score(y, pred)
    recall = recall_score(y,pred,average = 'micro',zero_division = 0)
    precision = precision_score(y,pred,average = 'micro',zero_division = 0)

    return roc_auc,recall,precision

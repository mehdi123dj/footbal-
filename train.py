from football_dataset import MyFootBallDataset 
from torch_geometric.loader.dataloader import DataLoader
from model import GAT_model,Transformer_model, train, test
import torch 
import warnings
import copy 
import os 
import argparse
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings('always') 
data_dir = '/home/mehdi/Desktop/Datarvest/Projects/Demo/football/data'
models_dir = '/home/mehdi/Desktop/Datarvest/Projects/Demo/football/models'


def main(): 

    """
    Collect arguments and run.
    """

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "-hd",
        "--hidden",
        default=128,
        type=int,
    )

    parser.add_argument(
        "-oc",
        "--out-channels",
        default=32,
        type=int,
    )

    parser.add_argument(
        "-nh",
        "--number-heads",
        default=1,
        type=int,
    )

    parser.add_argument(
        "-ne",
        "--number-epochs",
        default=1000,
        type=int,
    )

    parser.add_argument(
        "-bs",
        "--batch-size",
        default=8,
        type=int,
    )

    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=0.005,
        type=float,
    )

    parser.add_argument(
        "-t",
        "--type",
        default = "Transformer",
        type=str,
    )
    parser.add_argument(
        "-d",
        "--dropout",
        default = .33,
        type=float,
    )
    args = parser.parse_known_args()[0]

    parser.add_argument(
        "-mn",
        "--model-name",
        default = 'model_AML_'+args.type+'.pt',
        type=str,
    )
    args = parser.parse_args()
    writer = SummaryWriter()

    data_train = MyFootBallDataset(data_dir,split = 'train')
    data_val = MyFootBallDataset(data_dir,split = 'val')
    data_test = MyFootBallDataset(data_dir,split = 'test')
    

    if args.type ==  "Transformer":
        model = Transformer_model(
                                    in_channels_node = data_train[0].x.shape[1],
                                    in_channels_edges =  data_train[0].edge_attr.shape[1],
                                    hidden = args.hidden,
                                    out_channels = args.out_channels,
                                    heads = args.number_heads,
                                    num_classes = data_train.num_classes,
                                    dropout = args.dropout
                                )   
    elif args.type ==  "GAT":
        model = GAT_model(
                            in_channels_node = data_train[0].x.shape[1],
                            in_channels_edges =  data_train[0].edge_attr.shape[1],
                            hidden = args.hidden,
                            out_channels = args.out_channels,
                            heads = args.number_heads,
                            num_classes = data_train.num_classes,
                            dropout = args.dropout
                        )   
    else : 
        raise Exception("The chosen model is not implemented")

    train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(data_val, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(data_test, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)

    best_val_F1 = 0
    print('Start training model with', args.type,'type')
    for epoch in tqdm(range(1, args.number_epochs+1)):
        loss = train(model,train_loader,optimizer,device)
        val_roc_auc, val_recall, val_precision = test(model,val_loader,device)
        test_roc_auc, test_recall, test_precision = test(model,test_loader,device)

        F1_val = 2 * (val_precision * val_recall) / (val_precision + val_recall)
        F1_test = 2 * (test_precision * test_recall) / (test_precision + test_recall)

        if F1_val > best_val_F1:
            best_model = copy.deepcopy(model)
            best_val_F1 = F1_val
            best_test_F1 = F1_test

        writer.add_scalar('Loss', loss, epoch)

        writer.add_scalar('Val/recall', val_recall, epoch)
        writer.add_scalar('Val/precision', val_precision, epoch)
        writer.add_scalar('Val/F1', F1_val, epoch)

        writer.add_scalar('Test/recall', test_recall, epoch)
        writer.add_scalar('Test/precision', test_precision, epoch)
        writer.add_scalar('Test/F1', F1_test, epoch)


    writer.close()

    SAVEPATH = os.path.join(models_dir,args.model_name)
    print('Done training')
    torch.save(best_model, SAVEPATH)
    print(f'Final Test: {best_test_F1:.4f}')


if __name__ == "__main__":
    main()
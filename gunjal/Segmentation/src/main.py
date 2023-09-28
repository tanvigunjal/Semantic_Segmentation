import torch 
import argparse

from utils.model.unet import U_Net_model
from utils.model.attention_unet import AttU_Net_model
from utils.model.r2u_net import R2U_Net_Model
from utils.cityscapes_dataset import dataloader
from utils.trainer import train, validation
from utils.test import test
from utils.evaluation_metrics import val_metric

def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataloaders
    train_loader, val_loader, test_loader = dataloader(config)

    # model selection
    if config.model == "unet":
        model = U_Net_model(3, config.num_classes).to(device)
    elif config.model == "att_unet":
        model = AttU_Net_model(3, config.num_classes).to(device)
    elif config.model == "r2u_net":
        model = R2U_Net_Model(3, config.num_classes, t=2).to(device)
    else:
        raise NotImplementedError

    # loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # training
    if config.mode == "train":
        for epoch_i in range(config.epochs):
            losses = train(train_loader, model, optimizer, criterion, device, config, epoch_i)
            score = validation(val_loader, model, epoch_i, criterion, device, config)
            print(score)
            # save model 
            torch.save(model.state_dict(), f"{config.out_dir}/{config.model}_{epoch_i}.pth")
            val_metric(score)
    # testing
    elif config.mode == "test":
        best_model = " "
        test(test_loader, model, criterion, epoch_i, config.epochs)
    else:
        raise NotImplementedError 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or test ResNet model.")
    parser.add_argument("--dataset_path", type=str, default="/ds/images/Cityscapes/", help="dataset path")
    parser.add_argument("--model", type=str, default="unet", help="model name u_net or R2U_Net or AttU_Net or R2AttU_Net")
    parser.add_argument("--mode", type=str, default="train", help="train or test")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--num_workers", type=int, default=1, help="number of workers")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--epochs", type=int, default=1, help="number of epochs")
    parser.add_argument("--num_classes", type=int, default=19, help="number of classes")
    parser.add_argument("--out_dir", type=str, default="/netscratch/gunjal/SWATS", help="output directory")
    args = parser.parse_args()

    main(args)

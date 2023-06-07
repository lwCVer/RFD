import datetime
import os
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from ptflops import get_model_complexity_info
from utils import train_one_epoch, evaluate, create_lr_scheduler, read_train_data, read_val_data, MyDataSet


# Selecting a backbone network

# from backbones.gfnet import gfnet_tiny as create_model
# from backbones.gfnet import gfnet_small as create_model
# from backbones.gfnet import gfnet_base as create_model
# from backbones.gfnet_RFD import gfnet_tiny as create_model
# from backbones.gfnet_RFD import gfnet_small as create_model
# from backbones.gfnet_RFD import gfnet_base as create_model
# from backbones.asmlp import AS_MLP_tiny as create_model
# from backbones.asmlp import AS_MLP_small as create_model
# from backbones.asmlp import AS_MLP_base as create_model
# from backbones.asmlp_RFD import AS_MLP_tiny as create_model
# from backbones.asmlp_RFD import AS_MLP_small as create_model
# from backbones.asmlp_RFD import AS_MLP_base as create_model
# from backbones.cswin import CSWin_64_12211_tiny_224 as create_model
# from backbones.cswin import CSWin_64_24322_small_224 as create_model
# from backbones.cswin import CSWin_96_24322_base_224 as create_model
# from backbones.cswin_RFD import CSWin_64_12211_tiny_224 as create_model
# from backbones.cswin_RFD import CSWin_64_24322_small_224 as create_model
# from backbones.cswin_RFD import CSWin_96_24322_base_224 as create_model
# from backbones.mixformer import MixFormer_B0 as create_model
# from backbones.mixformer import MixFormer_B2 as create_model
# from backbones.mixformer import MixFormer_B4 as create_model
# from backbones.mixformer_RFD import MixFormer_B0 as create_model
# from backbones.mixformer_RFD import MixFormer_B2 as create_model
# from backbones.mixformer_RFD import MixFormer_B4 as create_model
# from backbones.convnext import convnext_tiny as create_model
# from backbones.convnext import convnext_small as create_model
# from backbones.convnext import convnext_base as create_model
# from backbones.convnext_RFD import convnext_tiny as create_model
# from backbones.convnext_RFD import convnext_small as create_model
# from backbones.convnext_RFD import convnext_base as create_model
# from backbones.swin_v2 import swin_tiny_patch4_window7_224 as create_model
# from backbones.swin_v2 import swin_small_patch4_window7_224 as create_model
# from backbones.swin_v2 import swin_base_patch4_window7_224 as create_model
# from backbones.swin_v2_RFD import swin_tiny_patch4_window7_224 as create_model
# from backbones.swin_v2_RFD import swin_small_patch4_window7_224 as create_model
from backbones.swin_v2_RFD import swin_base_patch4_window7_224 as create_model



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--backbone', type=str, default='swin_v2_tiny_RFD', help='save result')

    parser.add_argument('--num_classes', type=int, default=45)
    parser.add_argument('--epochs', type=int, default=300)

    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--wd', type=float, default=0.05)
    parser.add_argument('--seed', type=bool, default=True)
    # dataset root
    parser.add_argument('--train-data-path', type=str, default="/data/dataset/Classification/NWPU-RESISC45/train/")
    parser.add_argument('--val-data-path', type=str, default="/data/dataset/Classification/NWPU-RESISC45/val/")
    # pretrain
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    # resume
    parser.add_argument('--resume', type=bool, default=False)
    # freeze-layers
    parser.add_argument('--freeze-layers', type=bool, default=False)
    # device
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    args = parser.parse_args()
    return args


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")
    backbone = args.backbone
    time = "results_{}.txt".format(datetime.datetime.now().strftime("%y%m%d-%H%M"))
    output = os.path.join('./output', backbone)
    results_file = os.path.join(output, time)
    if os.path.exists(output) is False:
        os.makedirs(output)

    tensorboard = os.path.join(output, 'runs')
    tb_writer = SummaryWriter(tensorboard)

    if args.seed:
        seed = 1234
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    train_images_path, train_images_label = read_train_data(args.train_data_path)
    val_images_path, val_images_label = read_val_data(args.val_data_path)

    img_size = 224
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandAugment(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])


    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes).to(device)

    # flops and params
    flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
    print("params: ", params)
    print("flops: ", flops)

    with open(results_file, "a") as f:
        info = f"args: {args}\n"
        f.write(info + "\n")

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        for k in list(weights_dict.keys()):
            if "head" in k:
                print("delete:", k)
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            para.requires_grad_(False)

        for name, para in model.named_parameters():
            if "head" in name:
                para.requires_grad_(True)
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True, warmup_epochs=2)

    best_acc = 0.0
    best_epoch = 0

    start_epoch = -1
    if args.resume:
        path_checkpoint = "last-val_acc.pth"
        checkpoint = torch.load(path_checkpoint)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']

    with open(results_file, "a") as f:
        info = f"params: {params}\n" \
               f"flops: {flops}\n"
        f.write(info + "\n\n")


    for epoch in range(start_epoch + 1, args.epochs):
        # train
        model.train()
        train_loss, train_acc = train_one_epoch(model=model, optimizer=optimizer, scheduler=scheduler,
                                                data_loader=train_loader, device=device, epoch=epoch)
        # validate
        model.eval()
        val_loss, val_acc = evaluate(model=model, data_loader=val_loader,
                                     device=device, epoch=epoch)
        # save checkpoint
        save_path = os.path.join(output, "weights")
        if epoch >= 0:
            checkpoint = {
                "model": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch}
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            torch.save(checkpoint, './output/{}/weights/last-val_acc.pth'.format(backbone))

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            checkpoint = {
                "model": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch}
            torch.save(checkpoint, './output/{}/weights/best-val_acc.pth'.format(backbone))

        with open(results_file, "a") as f:
            info = f"[epoch: {epoch}]  "\
                   f"train_acc: {train_acc:.4f}  " \
                   f"train_loss: {train_loss:.4f}  " \
                   f"val_acc: {val_acc:.4f}  "\
                   f"val_loss: {val_loss:.4f}  "
            f.write(info + "\n")

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "best_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], best_acc, epoch)
        tb_writer.add_scalar(tags[5], optimizer.param_groups[0]["lr"], epoch)

    with open(results_file, "a") as f:
        info = f"best_epoch: {best_epoch}\n" \
               f"best_acc: {best_acc:.4f}"
        f.write(info + "\n\n")


if __name__ == '__main__':
    args = parse_args()

    main(args)

import os
import wandb
import torch
import torch.nn as nn
import json
import numpy as np
import math
import copy
from dataset import TrainOrchidDataset,ValOrchidDataset
from config import get_args
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
import pandas as pd
import os
from utils import SAM,enable_running_stats,smooth_crossentropy,disable_running_stats
import yaml

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        if param_group["lr"] is not None:
            return param_group["lr"]

def adjust_lr(iteration, optimizer, schedule):
    for param_group in optimizer.param_groups:
        param_group["lr"] = schedule[iteration]



def set_environment(args):

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
    )
    train_transforms = transforms.Compose([
                        transforms.Resize((600, 600), Image.BILINEAR),
                        transforms.CenterCrop((args.data_size, args.data_size)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1),
                        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
                        transforms.ToTensor(),
                        normalize
    ])
    val_transforms = transforms.Compose([
                        transforms.Resize((600, 600), Image.BILINEAR),
                        transforms.CenterCrop((args.data_size, args.data_size)),
                        transforms.ToTensor(),
                        normalize
    ])
    train_set = TrainOrchidDataset(args.data_root,train_transforms)
    test_set = ValOrchidDataset(args.data_root,val_transforms)
    train_loader = torch.utils.data.DataLoader(train_set, num_workers=args.num_workers, shuffle=True, batch_size=args.batch_size) 
    test_loader = torch.utils.data.DataLoader(test_set, num_workers=1, shuffle=False, batch_size=args.batch_size)

    print("train samples: {}, train batchs: {}".format(len(train_set), len(train_loader)))
    print("test samples: {}, test batchs: {}".format(len(test_set), len(test_loader)))
    
    if args.model_name == "efficientnet-b7":
        from models.EfficientNet_FPN import DetEfficientNet
        model = DetEfficientNet(in_size=args.data_size,
                                num_classes=args.num_classes, 
                                use_fpn=args.use_fpn, 
                                use_ori=args.use_ori,
                                use_gcn=args.use_gcn,
                                use_layers=args.use_layers,
                                use_selections=args.use_selections,
                                num_selects=args.num_selects,
                                global_feature_dim=args.global_feature_dim)
    elif args.model_name == 'resnet-50':
        from models.ResNet50_FPN import DetResNet50
        model = DetResNet50(in_size=args.data_size,
                            num_classes=args.num_classes, 
                            use_fpn=args.use_fpn, 
                            use_ori=args.use_ori,
                            use_gcn=args.use_gcn,
                            use_layers=args.use_layers,
                            use_selections=args.use_selections,
                            num_selects=args.num_selects,
                            global_feature_dim=args.global_feature_dim)
    elif args.model_name == 'vit-b16':
        from models.Vitb16_FPN import VitB16
        model = VitB16(in_size=args.data_size,
                       num_classes=args.num_classes, 
                       use_fpn=args.use_fpn, 
                       use_ori=args.use_ori,
                       use_gcn=args.use_gcn,
                       use_layers=args.use_layers,
                       use_selections=args.use_selections,
                       num_selects=args.num_selects,
                       global_feature_dim=args.global_feature_dim)
    elif args.model_name == 'swin-vit-p4w12':
        from models.SwinVit12 import SwinVit12
        model = SwinVit12(
                in_size=args.data_size,
                num_classes=args.num_classes, 
                use_fpn=args.use_fpn, 
                use_ori=args.use_ori,
                use_gcn=args.use_gcn,
                use_layers=args.use_layers,
                use_selections=args.use_selections,
                num_selects=args.num_selects,
                global_feature_dim=args.global_feature_dim
            )

    checkpoint = torch.load(args.pretrained_path)
    pretrained_dict = checkpoint['model']
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and pretrained_dict[k].shape == model_dict[k].shape}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    model.to(args.device)
    
    if args.optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=args.max_lr, 
                                    nesterov=args.nesterov, 
                                    momentum=0.9, 
                                    weight_decay=args.wdecay)
    elif args.optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_lr)
    elif args.optimizer_name == "Sam":
        base_optimizer = torch.optim.SGD
        optimizer = SAM(model.parameters(), base_optimizer, rho=2.0, adaptive=True, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # lr schedule
    total_batchs = args.max_epochs * len(train_loader)
    iters = np.arange(total_batchs - args.warmup_batchs)
    schedule = np.array([1e-12 + 0.5 * (args.max_lr - 1e-12) * (1 + \
                         math.cos(math.pi * t / total_batchs)) for t in iters])

    # schedule = args.max_lr * np.array([math.cos(7*math.pi*t / (16*total_batchs)) for t in iters])
    if args.warmup_batchs > 0:
        warmup_lr_schedule = np.linspace(1e-9, args.max_lr, args.warmup_batchs)
        schedule = np.concatenate((warmup_lr_schedule, schedule))


    return train_loader, test_loader, model, optimizer, schedule


def train(args, epoch, model, scaler, optimizer, schedules, train_loader, save_distrubution=False):

    model.train()

    optimizer.zero_grad()
    for batch_id, (datas, labels) in enumerate(train_loader):

        # adjust learning rate
        iterations = epoch * len(train_loader) + batch_id
        adjust_lr(iterations, optimizer, schedule)

        """ data preparation """
        # batch size (full)
        batch_size = labels.size(0)
       
        """ forward """
        datas, labels = datas.to(args.device), labels.to(args.device)

        if args.optimizer_name == "Sam":
            # first forward-backward step
            enable_running_stats(model)

        losses, accuracys = model(datas, labels)
        
        loss = 0
        for name in losses:
            if "selected" in name:
                loss += losses[name]
            if "ori" in name:
                loss += losses[name]
            else:
                loss += losses[name]
        
        loss /= args.update_freq
        

        if args.optimizer_name == "Sam":
            loss.backward()
            optimizer.first_step(zero_grad=True)
            # second forward-backward step
            disable_running_stats(model) 
            losses, accuracys = model(datas, labels)
            loss = 0
            for name in losses:
                if "selected" in name:
                    loss += losses[name]
                if "ori" in name:
                    loss += losses[name]
                else:
                    loss += losses[name]
            loss /= args.update_freq
            loss.backward()
            optimizer.second_step(zero_grad=True)
        else:
            scaler.scale(loss).backward()
            if (batch_id+1) % args.update_freq == 0:
                scaler.step(optimizer)
                scaler.update() # next batch.
                optimizer.zero_grad()

        
        """ log """
        if (batch_id+1) % args.log_freq == 0:
            msg = {
                "train_info/epoch":epoch+1,
                "train_loss/loss":loss,
                "train_info/lr":get_lr(optimizer)
            }
            for name in accuracys:
                msg["train_acc/train_acc_"+name] = 100*accuracys[name]

            for name in losses:
                msg["train_loss/train_loss_"+name] = losses[name]

            wandb.log(msg)

        


def test(args, model, test_loader):
    

    total = 0

    accuracys = {"sum":0}
    global_accs_template = {}
    for i in args.test_global_top_confs:
        global_accs_template["global_top"+str(i)] = 0
    
    select_accs_template = {}
    for i in args.test_select_top_confs:
        select_accs_template["select_top"+str(i)] = 0

    model.eval()
    with torch.no_grad():
        for batch_id, (datas, labels) in enumerate(test_loader):
            
            """ data preparation """
            batch_size = labels.size(0)
            total += batch_size

            datas, labels = datas.to(args.device), labels.to(args.device)

            """ forward """
            _, batch_accs, batch_logits = model(datas, labels, return_preds=True)
            
            for name in batch_accs:
                store_name = name
                if store_name not in  accuracys:
                    accuracys[store_name] = 0
                accuracys[store_name] += batch_accs[name]*batch_size

            labels = labels.cpu()
            
            # = = = = = output post-processing. = = = = = 
            # = = = softmax = = =
            for name in batch_logits:
                if name in ["ori"]:
                    batch_logits[name] = torch.softmax(batch_logits[name], dim=1)
                elif "l_" in name:
                    batch_logits[name] = torch.softmax(batch_logits[name].mean(2).mean(2), dim=-1)
                elif "select" in name:
                    batch_logits[name] = torch.softmax(batch_logits[name], dim=-1)
                elif name in ["gcn"]:
                    batch_logits[name] = torch.softmax(batch_logits[name], dim=-1)
                
                batch_logits[name] = batch_logits[name].cpu()

            """
            ori
            gcn
            layers
            selecteds (sorted)
            """
            # 1. ========= sum (average) =========
            logit_sum = None
            for name in batch_logits:
                # = = = skip = = =
                if "select" in name:
                    continue

                if logit_sum is None:
                    logit_sum = batch_logits[name]
                else:
                    logit_sum += batch_logits[name]

            accuracys["sum"] = torch.max(logit_sum, dim=-1)[1].eq(labels).sum().item()

            # 2. ========= vote =========
            pred_counter = torch.zeros([batch_size, args.num_classes])
            pred_counter_select = torch.zeros([batch_size, args.num_classes])
            for name in batch_logits:
                if "selected" in name:
                    """
                    [B, S, C]
                    """
                    preds = torch.max(batch_logits[name], dim=-1)[1]
                    for bid in range(batch_size):
                        batch_pred = preds[bid]
                        for pred in batch_pred:
                            pred_cls = pred.item()
                            pred_counter_select[bid][pred_cls] += 1
                    continue

                """
                [B, C]
                """
                preds = torch.max(batch_logits[name], dim=-1)[1]
                for bid in range(batch_size):
                    pred_cls = preds[bid]
                    pred_counter[bid][pred_cls] += 1
                    pred_counter_select[bid][pred_cls] += 1
            
            vote = torch.max(pred_counter, dim=-1)[1]
            vote_select = torch.max(pred_counter_select, dim=-1)[1]

            accuracys["vote"] = vote.eq(labels).sum().item()
            accuracys["vote_select"] = vote_select.eq(labels).sum().item()

            # 3. ========= bigger confidence prediction =========
            # 3.1 === global ===
            global_confidences = []
            # global_predictions = []
            global_features = []
            for name in batch_logits:
                if "select" in name:
                    continue
                confs, preds = torch.max(batch_logits[name], dim=-1)
                global_confidences.append(confs.unsqueeze(1))
                global_features.append(batch_logits[name].unsqueeze(1))

            global_confidences = torch.cat(global_confidences, dim=1) # B, S
            global_features = torch.cat(global_features, dim=1) # B, S, C

            area_size = global_confidences.size(1)

            # tmp variables.
            tmp_g_accs = copy.deepcopy(global_accs_template)
            # get batch acuracy
            for bid in range(batch_size):
                feature_sum = None
                ids = torch.sort(global_confidences[bid], dim=-1)[1] # S
                for i in range(args.test_global_top_confs[-1]):
                    if i >= ids.size(0):
                        break
                    fid = ids[i]
                    if feature_sum is None:
                        feature_sum = global_features[bid][fid]
                    else:
                        feature_sum += global_features[bid][fid]

                    if i in args.test_global_top_confs:
                        if torch.max(feature_sum, dim=-1)[1] == labels[bid]:
                            tmp_g_accs["global_top"+str(i)] += 1

            for name in tmp_g_accs:
                if name not in accuracys:
                    accuracys[name] = 0
                accuracys[name] += tmp_g_accs[name]

            # 3.2 === select ===
            tmp_s_accs = copy.deepcopy(select_accs_template)
            select_confs = []
            select_features = []
            for name in batch_logits:
                if "selected" not in name:
                    continue
                features = batch_logits[name] # [B, S, C]
                conf, pred = torch.max(features, dim=-1)
                select_confs.append(conf)
                select_features.append(features)
            
            if len(select_confs) > 0:
                select_confs = torch.cat(select_confs, dim=1)
                select_features = torch.cat(select_features, dim=1)

                # tmp variables.
                tmp_s_accs = copy.deepcopy(select_accs_template)
                # get batch acuracy
                for bid in range(batch_size):
                    feature_sum = None
                    ids = torch.sort(select_confs[bid], dim=-1)[1] # S
                    for i in range(args.test_select_top_confs[-1]):
                        if i>=ids.size(0):
                            break
                        fid = ids[i]
                        if feature_sum is None:
                            feature_sum = select_features[bid][fid]
                        else:
                            feature_sum += select_features[bid][fid]

                        if i in args.test_select_top_confs:
                            if torch.max(feature_sum, dim=-1)[1] == labels[bid]:
                                tmp_s_accs["select_top"+str(i)] += 1

                for name in tmp_s_accs:
                    if name not in accuracys:
                        accuracys[name] = 0
                    accuracys[name] += tmp_s_accs[name]

            # print(total, accuracys)

    # acc_final, acc_l1, acc_l2, acc_l3, acc_gcn
    best_acc = -1
    msg = {}
    for name in accuracys:
        msg["test_acc/test_acc_"+name] = 100*accuracys[name]/total
        if msg["test_acc/test_acc_"+name] > best_acc:
            best_acc = msg["test_acc/test_acc_"+name]
            
    msg["Current_Test_Acc"] = best_acc
    wandb.log(msg)

    return  best_acc



if __name__ == "__main__":
    args = get_args()
    
    wandb.init(project="Orchid",name=args.exp_name,config=args)

    train_loader, test_loader, model, optimizer, schedule = set_environment(args)

    scaler = torch.cuda.amp.GradScaler()

    best_acc = 0.0
    wandb.run.summary["best_accuracy"] = best_acc
    save_dist = False
    for epoch in range(args.max_epochs):
        print(f"Training Epoch:{epoch}")
        # control save distribution or not.
        if epoch == 0 or (epoch+1) % args.test_freq == 0:
            save_dist = True
        else:
            save_dist = False
        
        """ train model """
        train(args, epoch, model, scaler, optimizer, schedule, train_loader, save_distrubution=save_dist)

        # control test or not
        if epoch>args.max_epochs*0.9:
            args.test_freq = 1
        elif epoch>args.max_epochs*0.8:
            args.test_freq = 2
        elif epoch>args.max_epochs*0.6:
            args.test_freq = 4

        # save epoch model.
        save_dict = {'epoch': epoch}
        save_dict["model"] = model.state_dict()
        save_dict["optimizer"] = optimizer.state_dict()

        if epoch == 0 or (epoch+1) % args.test_freq == 0:
            test_acc = test(args, model, test_loader)
            # save to best.pt
            torch.save(save_dict, args.save_root + "last.pth")
            if test_acc > best_acc:
                best_acc = test_acc
                # save yaml
                with open(args.save_root +"config.yaml",'r') as yamlfile:
                    cur_yaml = yaml.safe_load(yamlfile)
                    if epoch == 0:
                        new_data = {'wandb': f"https://wandb.ai/angelowen/Orchid/runs/{wandb.run.id}/overview",'best_acc':best_acc} 
                        cur_yaml[-1]['info'] = (new_data)
                    else:
                        cur_yaml[-1]['info']['best_acc'] = best_acc
                with open(args.save_root +"config.yaml", 'w') as file:
                    documents = yaml.safe_dump(cur_yaml, file)
                    
                print("Best Acc: ", best_acc)
                wandb.run.summary["best_accuracy"] = best_acc # upload to wandb
                wandb.run.summary["best_epoch"] = epoch+1 # upload to wandb
                if os.path.isfile(args.save_root + "best.pth"):
                    os.remove(args.save_root + "best.pth")
                torch.save(save_dict, args.save_root + "best.pth")


import os
import torch
import torch.nn as nn
import json
import numpy as np
import math
import copy
from dataset import ValOrchidDataset
from config_eval import get_args
from torchvision import transforms
import tqdm
from PIL import Image
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def set_environment(args,path1,path2):

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
    )
    val_transforms = transforms.Compose([
        transforms.Resize((600, 600), Image.BILINEAR),
        transforms.CenterCrop((args.data_size, args.data_size)),
        transforms.ToTensor(),
        normalize
    ])

    test_set = ValOrchidDataset(args.data_root,val_transforms) # data_transform
    test_loader = torch.utils.data.DataLoader(test_set, num_workers=1, shuffle=False, batch_size=args.batch_size)

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
        model2 = SwinVit12(
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

    checkpoint = torch.load(path1)
    print(f"\n\nLoading checkpoint {path1}")
    print("Please check if it is the right position!!")
    model.load_state_dict(checkpoint['model'])
    model.to(args.device)
    
    checkpoint = torch.load(path2)
    print(f"\n\nLoading checkpoint2 {path2}")
    model2.load_state_dict(checkpoint['model'])
    model2.to(args.device)

    return test_loader, model, model2



def test(args, model,model2, test_loader):
    
    total = 0

    accuracys = {"sum":0}
    global_accs_template = {}
    for i in args.test_global_top_confs:
        global_accs_template["global_top"+str(i)] = 0

    pbar = tqdm.tqdm(total=len(test_loader), ascii=True)

    model.eval()
    model2.eval()
    with torch.no_grad():
        for batch_id, (datas, labels) in enumerate(test_loader):
            
            """ data preparation """
            batch_size = labels.size(0)
            total += batch_size

            datas, labels = datas.to(args.device), labels.to(args.device)
            _, batch_accs, batch_logits = model(datas, labels, return_preds=True)
            _, batch_accs2, batch_logits2 = model2(datas, labels, return_preds=True)

            """ forward """        
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
                    batch_logits2[name] = torch.softmax(batch_logits2[name], dim=1)
                elif "l_" in name:
                    batch_logits[name] = torch.softmax(batch_logits[name].mean(2).mean(2), dim=-1)
                    batch_logits2[name] = torch.softmax(batch_logits2[name].mean(2).mean(2), dim=-1)
                elif "select" in name:
                    batch_logits[name] = torch.softmax(batch_logits[name], dim=-1)
                    batch_logits2[name] = torch.softmax(batch_logits2[name], dim=-1)
                elif name in ["gcn"]:
                    batch_logits[name] = torch.softmax(batch_logits[name], dim=-1)
                    batch_logits2[name] = torch.softmax(batch_logits2[name], dim=-1)
                batch_logits[name] = batch_logits[name].cpu()
                batch_logits2[name] = batch_logits2[name].cpu()

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

            # 2. ========= bigger confidence prediction =========
            # 3.1 === global ===
            global_confidences = []
            global_confidences2 = []
            # global_predictions = []
            global_features = []
            global_features2 = []
            for name in batch_logits:
                if "select" in name:
                    continue
                confs, preds = torch.max(batch_logits[name], dim=-1)
                global_confidences.append(confs.unsqueeze(1))
                global_features.append(batch_logits[name].unsqueeze(1))
 
                confs2, preds2 = torch.max(batch_logits2[name], dim=-1)
                global_confidences2.append(confs2.unsqueeze(1))
                global_features2.append(batch_logits2[name].unsqueeze(1))
                # print(confs.shape,confs2.shape)
            

            global_confidences = torch.cat(global_confidences, dim=1) # B, S
            global_features = torch.cat(global_features, dim=1) # B, S, C
            
            global_confidences2 = torch.cat(global_confidences2, dim=1) # B, S
            global_features2 = torch.cat(global_features2, dim=1) # B, S, C

            # tmp variables.
            tmp_g_accs = copy.deepcopy(global_accs_template)
            
            # eval sample in batch
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
                        # logits1 = torch.softmax(feature_sum,dim=-1)
                        logits1 = feature_sum
                        # if torch.max(feature_sum, dim=-1)[1] == labels[bid]:
                        #     tmp_g_accs["global_top"+str(i)] += 1
                # ----------------------------------
                feature_sum = None
                ids = torch.sort(global_confidences2[bid], dim=-1)[1] # S
                for i in range(args.test_global_top_confs[-1]):
                    if i >= ids.size(0):
                        break
                    fid = ids[i]                   
                    if feature_sum is None:
                        feature_sum = global_features2[bid][fid]
                    else:
                        feature_sum += global_features2[bid][fid]

                    if  i in args.test_global_top_confs:
                        # logits2 = torch.softmax(feature_sum,dim=-1)
                        logits2 = feature_sum
                        logits = 0.51*logits1 + 0.49*logits2 
                        if torch.max(logits, dim=-1)[1] == labels[bid]:
                            tmp_g_accs["global_top"+str(i)] += 1

            for name in tmp_g_accs:
                if name not in accuracys:
                    accuracys[name] = 0
                accuracys[name] += tmp_g_accs[name]

            pbar.update(1)

    pbar.close()
    max_acc = -1
    msg = ""
    for name in accuracys:
        acc = 100*accuracys[name]/total
        acc = round(acc, 3)
        if acc>max_acc:
            max_acc = acc

        msg += "acc_" + name + ":" + str(acc) + "\n"
    
    print()
    print(msg)
    print("\n\nbest: {}%\n".format(max_acc))

if __name__ == "__main__":
    args = get_args()
    test_loader, model,model2 = set_environment(args,'backup/best_94_97.pth','backup/best_94_06.pth')
    test(args, model,model2, test_loader)

import os
import torch
import torch.nn as nn
import json
import numpy as np
import math
import copy
from dataset import TrainOrchidDataset,ValOrchidDataset
from config_eval import get_args
from torchvision import transforms
import pandas as pd
import tqdm
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def set_environment(args):

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
    data_transform = transforms.Compose([
        transforms.CenterCrop(args.data_size),
        transforms.ToTensor(),
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

    checkpoint = torch.load(args.pretrained_path)
    model.load_state_dict(checkpoint['model'])
    model.to(args.device)

    return test_loader, model



def test(args, model, test_loader):
    pred_layer = 3 ############## output layer prediction
    print(f"Output the prediction on Layer acc_global_top{pred_layer}")
    total = 0
    answer = []
    accuracys = {"sum":0}
    global_accs_template = {}
    for i in args.test_global_top_confs:
        global_accs_template["global_top"+str(i)] = 0

    pbar = tqdm.tqdm(total=len(test_loader), ascii=True)

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

                        if i == pred_layer:
                            preds = torch.max(feature_sum, dim=-1)[1]
                            answer.append(preds.item())
                        if torch.max(feature_sum, dim=-1)[1] == labels[bid]:
                            tmp_g_accs["global_top"+str(i)] += 1

            for name in tmp_g_accs:
                if name not in accuracys:
                    accuracys[name] = 0
                accuracys[name] += tmp_g_accs[name]

            pbar.update(1)

    pbar.close()
    
    output_file = "./Final_submission.csv"
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
    else:   
        df = pd.read_csv('dataset/val_label.csv')
    exist_voter = len(df.columns)-2
    df[f'voter_{exist_voter+1}'] = answer
    df.to_csv(output_file, index=False)
    print("finished!!")

    


if __name__ == "__main__":
    args = get_args()
    if args.tv:
        answer = []
        correct = 0
        df = pd.read_csv('./Final_submission.csv')
        num = len(df['filename'])
        for idx in range(num):
            ans = df.iloc[idx][2:].mode()[0] # start from voter_1
            answer.append(ans)
        '''     # Get Accuracy in following lines
            # ans1 = df['voter_1'][idx]
            # if ans != ans1:
            #     print(idx+2)
            if ans == df['category'][idx]:
                correct+=1
        print(correct/num)
        ''' 
        output_file = "./Vote_Final_submission.csv"
        df = pd.read_csv('dataset/val_label.csv')
        df['category'] = answer
        df.to_csv(output_file, index=False)
        print("finished!!")       
    else:    
        test_loader, model = set_environment(args)
        test(args, model, test_loader)

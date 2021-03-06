import argparse
import os

def get_args():

    parser = argparse.ArgumentParser("")
    checkpoint = ""
    if os.path.exists("./records/"):
        checkpoint_L = ["./records/"+name for name in os.listdir("./records") if os.path.isdir("./records/"+name)]
        checkpoint = max(checkpoint_L, key=os.path.getctime) 
    parser.add_argument("--pretrained_path", default=f"{checkpoint}/best.pth", type=str)
    parser.add_argument("--data_root", default="./dataset", type=str) 
    parser.add_argument("--data_size", default=384, type=int)
    parser.add_argument("--num_rows", default=0, type=int)
    parser.add_argument("--num_cols", default=0, type=int)
    parser.add_argument("--sub_data_size", default=32, type=int)

    parser.add_argument("--model_name", default="swin-vit-p4w12", type=str, 
        choices=["efficientnet-b7", 'resnet-50', 'vit-b16', 'swin-vit-p4w12'])
    parser.add_argument("--optimizer_name", default="sgd", type=str, 
        choices=["sgd", 'adamw'])
    
    parser.add_argument("--use_fpn", default=True, type=bool)
    parser.add_argument("--use_ori", default=False, type=bool)
    parser.add_argument("--use_gcn", default=True, type=bool)
    parser.add_argument("--use_layers", 
        default=[True, True, True, True], type=list)
    parser.add_argument("--use_selections", 
        default=[True, True, True, True], type=list)
    # [2048, 512, 128, 32] for CUB200-2011
    # [256, 128, 64, 32] for NABirds
    parser.add_argument("--num_selects",
        default=[2048, 512, 128, 32], type=list)
    parser.add_argument("--global_feature_dim", default=1536, type=int)
    
    # loader
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    
    # about model building
    parser.add_argument("--num_classes", default=219, type=int)
    parser.add_argument("--test_global_top_confs", default=[1,3,5], type=list)
    parser.add_argument("--tta",default=False, type=bool)
    parser.add_argument("--vote",help = 'test with voting after csv is done',default=False, action='store_true')
    parser.add_argument("--tv",help = 'test with voting after csv is done',default=False, action='store_true')
    args = parser.parse_args()

    return args

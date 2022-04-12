import argparse
import os
import time
import yaml
def get_args():
    
    t = time.localtime()
    run_time = time.strftime("%Y_%m_%d_%H_%M_%S", t)
    
    parser = argparse.ArgumentParser("FineGrained Image Classification Task")
    # save path and dataset information
    parser.add_argument("--exp_name", default=f"{run_time}")
    
    parser.add_argument("--data_root", default="./dataset", type=str) 
    parser.add_argument("--data_size", default=384, type=int)
    parser.add_argument("--num_rows", default=0, type=int)
    parser.add_argument("--num_cols", default=0, type=int)
    parser.add_argument("--sub_data_size", default=32, type=int)

    parser.add_argument("--model_name", default="swin-vit-p4w12", type=str, 
        choices=["efficientnet-b7", 'resnet-50', 'vit-b16', 'swin-vit-p4w12'])
    parser.add_argument("--optimizer_name", default="sgd", type=str, 
        choices=["sgd", 'adamw','Sam'])
    
    parser.add_argument("--use_fpn", default=True, type=bool)
    parser.add_argument("--use_ori", default=False, type=bool)
    parser.add_argument("--use_gcn", default=True, type=bool)
    parser.add_argument("--use_layers", 
        default=[True, True, True, True], type=list)
    parser.add_argument("--use_selections", 
        default=[True, True, True, True], type=list)
    parser.add_argument("--num_selects",
        default=[2048, 512, 128, 32], type=list)
    parser.add_argument("--global_feature_dim", default=1536, type=int)
    
    # loader
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    
    # about model building
    parser.add_argument("--num_classes", default=219, type=int)
    
    # abput learning rate scheduler
    parser.add_argument("--warmup_batchs", default=800, type=int)
    parser.add_argument("--no_final_epochs", default=0, type=int)
    parser.add_argument("--max_lr", default=0.0005, type=float)
    parser.add_argument("--update_freq", default=4, type=int)
    
    parser.add_argument("--wdecay", default=0.0005, type=float)
    parser.add_argument("--nesterov", default=True, type=bool)
    parser.add_argument("--max_epochs", default=60, type=int)

    parser.add_argument("--log_freq", default=10, type=int)

    parser.add_argument("--test_freq", default=3, type=int)
    parser.add_argument("--test_global_top_confs", default=[1, 3, 5], type=list)
    parser.add_argument("--test_select_top_confs", default=[1, 3, 5, 7, 9], type=list)
    parser.add_argument("--pretrained_path", default='./backup/pretrained.pth', type=str)

    args = parser.parse_args()
    args = build_record_folder(args)

    return args


def build_record_folder(args):
    print("building records folder...", end="")
    
    if not os.path.isdir("./records/"):
        print(".../records/...", end="")
        os.mkdir("./records/")
    args.save_root = "./records/" + args.exp_name + "/"
    os.makedirs(args.save_root, exist_ok=True)
    
    print("...{}...".format(args.save_root), end="")
    os.makedirs(args.save_root, exist_ok=True)


    dict_file = [{ 'Experiment Time': args.exp_name,
                'Data_size': args.data_size,
                'Sub_data_size': args.sub_data_size,
                'Model_name': args.model_name,
                'Optimizer': args.optimizer_name,
                'Num_selects':args.num_selects,
                'Global_feature_dim': args.global_feature_dim,
                'warmup_batchs': args.warmup_batchs,
                'Max_LR':args.max_lr,
                'Update_freq':args.update_freq,
                'Weight_Decay': args.wdecay,
                'Max epochs':args.max_epochs,
                }]

    with open(f'./{args.save_root}config.yaml', 'w') as file:
        documents = yaml.dump(dict_file, file)

    print("...finish")
    print()
    return args

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import models
from flTrainer import *
import copy
from model import *
import torchvision
from model.vgg import get_vgg_model
from model.resnet import ResNet18,ResNet50
from model.dla import DLA
from model.mobilenet import MobileNet
from model.googlenet import GoogLeNet

from torchsummary import summary

def bool_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='parameter board')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.00036, metavar='LR', help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.998, metavar='M', help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1234, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--local_training_epoch', type=int, default=1, help='number of local training epochs')
    parser.add_argument('--malicious_local_training_epoch', type=int, default=1, help='number of malicious local training epochs')
    parser.add_argument('--num_nets', type=int, default=200, help='number of totally available users')
    parser.add_argument('--part_nets_per_round', type=int, default=30, help='number of participating clients per FL round')
    parser.add_argument('--fl_round', type=int, default=100, help='total number of FL round to conduct')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to set, can take the value of: cuda or cuda:x')
    parser.add_argument('--dataname', type=str, default='cifar10', help='dataset to use during the training process')
    parser.add_argument('--num_class', type=int, default=10, help='number of classes for dataset')
    parser.add_argument('--datadir', type=str, default='./dataset/', help='the directory of dataset')
    parser.add_argument('--partition_strategy', type=str, default='hetero-dir', help='dataset iid(homo) or non-iid(hetero-dir)')
    parser.add_argument('--dir_parameter', type=float, default=0.5, help='the parameter of dirichlet distribution')
    parser.add_argument('--model', type=str, default='vgg9', help='model to use during the training process')
    parser.add_argument('--load_premodel', type=bool_string, default=True, help='whether load the pre-model in begining')
    parser.add_argument('--save_model', type=bool_string, default=False, help='whether save the intermediate model')
    parser.add_argument('--client_select', type=str, default='fix-frequency', help='the strategy for PS to select client: fix-frequency|fix-pool')

    # parameters for backdoor attacker
    parser.add_argument('--malicious_ratio', type=float, default=0.2, help='the ratio of malicious clients')
    parser.add_argument('--trigger_label', type=int, default=0, help='The NO. of trigger label (int, range from 0 to 9, default: 0)')
    parser.add_argument('--semantic_label', type=int, default=2, help='The NO. of semantic label (int, range from 0 to 9, default: 2)')
    parser.add_argument('--poisoned_portion', type=float, default=0.3, help='posioning portion (float, range from 0 to 1, default: 0.1)')
    parser.add_argument('--backdoor_type', type=str, default="none", help='backdoor type used: none|trigger|semantic|edge-case|')

    # parameters for defenders
    parser.add_argument('--defense_method', type=str, default="none",help='defense method used: none|krum|multi-krum|xmam|ndc|rsa|rfa|')
    parser.add_argument('--cut', type=int, default=60,help='defense method used: none|krum|multi-krum|xmam|ndc|rsa|rfa|')
    parser.add_argument('--test', type=bool_string, default="False", help='test model')

    #############################################################################
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': False} if use_cuda else {}
    device = torch.device(args.device if use_cuda else "cpu")

    torch.manual_seed(args.seed)
    criterion = nn.CrossEntropyLoss()
    ###################################################################################### select networks
    if args.model == "lenet":
        if args.load_premodel==True:
            net_avg = LeNet().to(device)
            with open("savedModel/mnist_.pt", "rb") as ckpt_file:
                ckpt_state_dict = torch.load(ckpt_file, map_location=device)
            net_avg.load_state_dict(ckpt_state_dict)
            if args.test:
                net_avg1 = LeNet().to(device)
                with open("savedModel/mnist_poi.pt", "rb") as ckpt_file:
                    ckpt_state_dict = torch.load(ckpt_file, map_location=device)
                net_avg1.load_state_dict(ckpt_state_dict)
                net_avg2=copy.deepcopy(net_avg)
                net_avg3=copy.deepcopy(net_avg1)
                whole_aggregator1 = []
                for param_index, p in enumerate(net_avg.parameters()):
                    whole_aggregator1.append(p.data)
                whole_aggregator2 = []
                for param_index, p in enumerate(net_avg1.parameters()):
                    whole_aggregator2.append(p.data)

                for param_index, p in enumerate(net_avg2.parameters()):
                    if param_index > args.cut:
                        p.data = whole_aggregator1[param_index]
                        continue
                    p.data = whole_aggregator2[param_index]

                for param_index, p in enumerate(net_avg3.parameters()):
                    if param_index > args.cut:
                        p.data = whole_aggregator2[param_index]
                        continue
                    p.data = whole_aggregator1[param_index]
            logger.info("Loading pre-model successfully ...")
        else:
            net_avg = LeNet().to(device)

    elif args.model in ("vgg9", "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn"):
        if args.load_premodel==True:
            net_avg = get_vgg_model(args.model, args.num_class).to(device)
            with open("savedModel/{}_.pt".format(args.dataname), "rb") as ckpt_file:
                ckpt_state_dict = torch.load(ckpt_file, map_location=device)
            net_avg.load_state_dict(ckpt_state_dict)
            logger.info("Loading pre-model successfully ...")
        else:
            net_avg = get_vgg_model(args.model, args.num_class).to(device)

    elif args.model in ("resnet18"):
        net_avg= models.resnet18(pretrained=True).to(device)
        net_avg.avgpool = nn.AdaptiveAvgPool2d(1).to(device)
        num_ftrs = net_avg.fc.in_features
        net_avg.fc = nn.Linear(num_ftrs, args.num_class).to(device)
        if args.load_premodel==True:
            with open("savedModel/{}_.pt".format(args.dataname,args.model), "rb") as ckpt_file:
                ckpt_state_dict = torch.load(ckpt_file, map_location=device)
            net_avg.load_state_dict(ckpt_state_dict)
            logger.info("Loading pre-model successfully ...")
        if args.test :
            net_avg1 = models.resnet18(pretrained=True).to(device)
            net_avg1.avgpool = nn.AdaptiveAvgPool2d(1).to(device)
            num_ftrs = net_avg1.fc.in_features
            net_avg1.fc = nn.Linear(num_ftrs, args.num_class).to(device)
            if args.load_premodel == True:
                with open("savedModel/{}_poi.pt".format(args.dataname, args.model), "rb") as ckpt_file:
                    ckpt_state_dict = torch.load(ckpt_file, map_location=device)
                net_avg1.load_state_dict(ckpt_state_dict)
                logger.info("Loading pre-model successfully ...")
            net_avg2 = models.resnet18(pretrained=True).to(device)
            net_avg2.avgpool = nn.AdaptiveAvgPool2d(1).to(device)
            num_ftrs = net_avg2.fc.in_features
            net_avg2.fc = nn.Linear(num_ftrs, args.num_class).to(device)
            net_avg3 = models.resnet18(pretrained=True).to(device)
            net_avg3.avgpool = nn.AdaptiveAvgPool2d(1).to(device)
            num_ftrs = net_avg3.fc.in_features
            net_avg3.fc = nn.Linear(num_ftrs, args.num_class).to(device)
            net_avg2 = copy.deepcopy(net_avg)
            net_avg3 = copy.deepcopy(net_avg1)
            net_avg2.fc.load_state_dict(net_avg1.fc.state_dict())
            net_avg3.fc.load_state_dict(net_avg.fc.state_dict())

    elif args.model =='mobilenet':
        if args.load_premodel==True:
            net_avg = MobileNet(args.num_class).to(device)
            with open("savedModel/{}_.pt".format(args.dataname), "rb") as ckpt_file:
                ckpt_state_dict = torch.load(ckpt_file, map_location=device)
            net_avg.load_state_dict(ckpt_state_dict)
            logger.info("Loading pre-model successfully ...")
        else:
            net_avg = MobileNet(args.num_class).to(device)
        net_avg2=MobileNet().to(device)
        net_avg3= MobileNet().to(device)
    for index,(name,p) in enumerate(net_avg.named_parameters()):
        print(str(index)+" "+name)
    ############################################################################ adjust data distribution
    if args.backdoor_type in ('none', 'trigger'):
        net_dataidx_map = partition_data(args.dataname, './dataset', args.partition_strategy, args.num_nets,
                                         args.dir_parameter,args.num_class)

    elif args.backdoor_type == 'semantic' and args.dataname =="cifar10":
        net_dataidx_map = partition_data_semantic(args.dataname, './dataset', args.partition_strategy, args.num_nets,
                                         args.dir_parameter)
    elif args.backdoor_type == 'edge-case' and args.dataname == "cifar10":
        net_dataidx_map = partition_data(args.dataname, './dataset', args.partition_strategy, args.num_nets,
                                         args.dir_parameter,args.num_class)
    else:
        logger.info("wrong backdoor type")
        sys.exit()

    ########################################################################################## load dataset
    train_data, test_data = load_init_data(dataname=args.dataname, datadir=args.datadir)

    ######################################################################################### create data loader
    if args.backdoor_type == 'none':
        test_data_ori_loader, _ = create_test_data_loader(args.dataname, test_data, args.trigger_label,
                                                    args.batch_size)
        test_data_backdoor_loader = test_data_ori_loader
    elif args.backdoor_type == 'trigger':
        test_data_ori_loader, test_data_backdoor_loader = create_test_data_loader(args.dataname, test_data, args.trigger_label,
                                                     args.batch_size)
    elif args.backdoor_type == 'semantic'and args.dataname =="cifar10":
        with open('./backdoorDataset/green_car_transformed_test.pkl', 'rb') as test_f:
            saved_greencar_dataset_test = pickle.load(test_f)

        logger.info("Backdoor (Green car) test-data shape we collected: {}".format(saved_greencar_dataset_test.shape))
        sampled_targets_array_test = args.semantic_label * np.ones((saved_greencar_dataset_test.shape[0],), dtype=int)  # green car -> label as bird

        semantic_testset = copy.deepcopy(test_data)
        semantic_testset.data = saved_greencar_dataset_test
        semantic_testset.targets = sampled_targets_array_test

        test_data_ori_loader, test_data_backdoor_loader = create_test_data_loader_semantic(test_data, semantic_testset,
                                                                                           args.batch_size)

    elif args.backdoor_type == 'edge-case'and args.dataname =="cifar10":
        with open('./backdoorDataset/southwest_images_new_test.pkl', 'rb') as test_f:
            saved_greencar_dataset_test = pickle.load(test_f)

        logger.info("Backdoor (Green car) test-data shape we collected: {}".format(saved_greencar_dataset_test.shape))
        sampled_targets_array_test = 9 * np.ones((saved_greencar_dataset_test.shape[0],), dtype=int)  # southwest airplane -> label as truck

        semantic_testset = copy.deepcopy(test_data)
        semantic_testset.data = saved_greencar_dataset_test
        semantic_testset.targets = sampled_targets_array_test

        test_data_ori_loader, test_data_backdoor_loader = create_test_data_loader_semantic(test_data, semantic_testset,args.batch_size)
    else:
        logger.info("wrong backdoor type")
        sys.exit()
    logger.info("Test the model performance on the entire task before FL process ... ")
    overall_acc = test_model(net_avg, test_data_ori_loader, device, print_perform=False)
    logger.info("Test the model performance on the backdoor task before FL process ... ")
    backdoor_acc = test_model(net_avg, test_data_backdoor_loader, device, print_perform=False)
    logger.info("=====Main task test accuracy=====: {}".format(overall_acc))
    logger.info("=====Backdoor task test accuracy=====: {}".format(backdoor_acc))
    if args.test:

        ma=[]
        ba=[]

        ma.append(overall_acc)
        ba.append(backdoor_acc)
        logger.info("Test the model performance on the entire task before FL process ... ")
        overall_acc = test_model(net_avg1, test_data_ori_loader, device, print_perform=False)
        logger.info("Test the model performance on the backdoor task before FL process ... ")
        backdoor_acc = test_model(net_avg1, test_data_backdoor_loader, device, print_perform=False)
        logger.info("=====Main task test accuracy=====: {}".format(overall_acc))
        logger.info("=====Backdoor task test accuracy=====: {}".format(backdoor_acc))
        ma.append(overall_acc)
        ba.append(backdoor_acc)
        logger.info("Test the model performance on the entire task before FL process ... ")
        overall_acc = test_model(net_avg2, test_data_ori_loader, device, print_perform=False)
        logger.info("Test the model performance on the backdoor task before FL process ... ")
        backdoor_acc = test_model(net_avg2, test_data_backdoor_loader, device, print_perform=False)
        logger.info("=====Main task test accuracy=====: {}".format(overall_acc))
        logger.info("=====Backdoor task test accuracy=====: {}".format(backdoor_acc))
        ma.append(overall_acc)
        ba.append(backdoor_acc)
        logger.info("Test the model performance on the entire task before FL process ... ")
        overall_acc = test_model(net_avg3, test_data_ori_loader, device, print_perform=False)
        logger.info("Test the model performance on the backdoor task before FL process ... ")
        backdoor_acc = test_model(net_avg3, test_data_backdoor_loader, device, print_perform=False)
        ma.append(overall_acc)
        ba.append(backdoor_acc)

    logger.info("=====Main task test accuracy=====: {}".format(overall_acc))
    logger.info("=====Backdoor task test accuracy=====: {}".format(backdoor_acc))

    if args.test :
        logger.info("=====Main task test accuracy=====: FC{}----F'C'{}----FC'{}----F'C{}".format(ma[0],ma[1],ma[2],ma[3]))
        logger.info("=====Backdoor task test accuracy=====: FC{}----F'C'{}----FC'{}----F'C{}".format(ba[0],ba[1],ba[2],ba[3]))
        sys.exit()
    arguments = {
        "net_avg": net_avg,
        "partition_strategy": args.partition_strategy,
        "dir_parameter": args.dir_parameter,
        "net_dataidx_map": net_dataidx_map,
        "num_nets": args.num_nets,
        "dataname": args.dataname,
        "num_class": args.num_class,
        "datadir": args.datadir,
        "model": args.model,
        "load_premodel":args.load_premodel,
        "save_model":args.save_model,
        "client_select":args.client_select,
        "part_nets_per_round": args.part_nets_per_round,
        "fl_round": args.fl_round,
        "local_training_epoch": args.local_training_epoch,
        "malicious_local_training_epoch": args.malicious_local_training_epoch,
        "args_lr": args.lr,
        "args_gamma": args.gamma,
        "batch_size": args.batch_size,
        "device": device,
        "test_data_ori_loader": test_data_ori_loader,
        "test_data_backdoor_loader": test_data_backdoor_loader,
        "malicious_ratio": args.malicious_ratio,
        "trigger_label": args.trigger_label,
        "semantic_label": args.semantic_label,
        "poisoned_portion": args.poisoned_portion,
        "backdoor_type": args.backdoor_type,
        "defense_method": args.defense_method,
        "cut":args.cut,
    }

    fl_trainer = FederatedLearningTrainer(arguments=arguments)
    fl_trainer.run()

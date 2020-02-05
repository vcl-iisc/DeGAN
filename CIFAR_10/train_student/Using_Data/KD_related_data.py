# Training a Student network on CIFAR-10 
# Teacher architecture: AlexNet, Student architecture: AlexNet half
from __future__ import print_function
import argparse
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from alexnet import AlexNet
from alexnet_half import AlexNet_half
from torch.utils.data.sampler import SubsetRandomSampler
from tensorboardX import SummaryWriter
import numpy as np

# CUDA_VISIBLE_DEVICES=0 python KD_related_data.py --batch-size 2048 --test-batch-size 1000 --epochs 5000 --lr 0.001 --seed 108 --log-interval 10 --temp 20 --lambda_ 1

writer = SummaryWriter()
if not os.path.exists("models"):
    os.makedirs("models")

def train(args, model, netS, device, train_loader, optimizer, epoch, temp, inc_classes):
    model.eval()
    netS.train()
    loss_all_sum = 0
    tot = 0
    teacher_student_correct_sum = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = torch.from_numpy(data.numpy()[np.isin(target,inc_classes)]).to(device)
        if data.shape[0] == 0:
            continue
        tot += data.shape[0]
        optimizer.zero_grad()
        data = data*2 - 1
        output_teacher_logits = model(data)
        output_student_logits = netS(data)
        output_teacher_logits_ht = output_teacher_logits / temp
        output_student_logits_ht = output_student_logits / temp
        sm_teacher_ht = F.softmax(output_teacher_logits_ht,dim=1)
        sm_student_ht = F.softmax(output_student_logits_ht,dim=1)
        sm_teacher = F.softmax(output_teacher_logits, dim=1)
        sm_student = F.softmax(output_student_logits, dim=1)
        loss_kd = nn.KLDivLoss(reduction='sum')(F.log_softmax(output_student_logits_ht, dim=1),F.softmax(output_teacher_logits_ht, dim=1))
        pred_class_argmax_teacher = sm_teacher.max(1, keepdim=True)[1]
        loss_ce = F.cross_entropy(output_student_logits, pred_class_argmax_teacher.view(data.shape[0]),reduction='sum')
        loss_all = args.lambda_*temp*temp*loss_kd + (1-args.lambda_)*loss_ce
        loss_all.backward()
        loss_all_sum += loss_all
        pred_class_argmax_student = sm_student.max(1, keepdim=True)[1]
        pred_class_argmax_teacher = pred_class_argmax_teacher.view(sm_teacher.shape[0])
        pred_class_argmax_student = pred_class_argmax_student.view(sm_teacher.shape[0])
        teacher_student_correct = torch.sum(pred_class_argmax_student==pred_class_argmax_teacher)
        teacher_student_correct_sum = teacher_student_correct_sum + (teacher_student_correct).cpu().data.numpy()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, int((batch_idx+1) * len(data)), int(len(train_loader.dataset)*0.8),
                100. * (batch_idx+1) / len(train_loader), (loss_all/data.shape[0]).item()))
    loss_all_mean = loss_all_sum / tot
    teacher_student_acc = 100. * teacher_student_correct_sum / tot
    print('Train set: Average loss: {:.4f}, Teacher-Student Accuracy: {}/{} ({:.0f}% )'.format(
        loss_all_mean, teacher_student_correct_sum, tot, teacher_student_acc))
    torch.save(netS.state_dict(), "models/"+str(epoch)+".pth")
    return loss_all_mean, teacher_student_acc

def val(args, model, netS, device, test_loader, epoch, val_test, temp, inc_classes):
    model.eval()
    netS.eval()
    loss_all_sum = 0
    tot = 0
    teacher_student_correct_sum = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if data.shape[0] == 0:
                continue
            data = torch.from_numpy(data.numpy()[np.isin(target,inc_classes)]).to(device)
            tot += data.shape[0]
            data = data*2 - 1
            output_teacher_logits = model(data)
            output_student_logits = netS(data)
            output_teacher_logits_ht = output_teacher_logits / temp
            output_student_logits_ht = output_student_logits / temp
            sm_teacher_ht = F.softmax(output_teacher_logits_ht,dim=1)
            sm_student_ht = F.softmax(output_student_logits_ht,dim=1)
            sm_teacher = F.softmax(output_teacher_logits, dim=1)
            sm_student = F.softmax(output_student_logits, dim=1)
            loss_kd = nn.KLDivLoss(reduction='sum')(F.log_softmax(output_student_logits_ht, dim=1),F.softmax(output_teacher_logits_ht, dim=1))
            pred_class_argmax_teacher = sm_teacher.max(1, keepdim=True)[1]
            loss_ce = F.cross_entropy(output_student_logits, pred_class_argmax_teacher.view(data.shape[0]),reduction='sum')
            loss_all = args.lambda_*temp*temp*loss_kd + (1-args.lambda_)*loss_ce
            loss_all_sum += loss_all
            pred_class_argmax_student = sm_student.max(1, keepdim=True)[1]
            pred_class_argmax_teacher = pred_class_argmax_teacher.view(sm_teacher.shape[0])
            pred_class_argmax_student = pred_class_argmax_student.view(sm_teacher.shape[0])
            teacher_student_correct = torch.sum(pred_class_argmax_student==pred_class_argmax_teacher)
            teacher_student_correct_sum = teacher_student_correct_sum + (teacher_student_correct).cpu().data.numpy()
    loss_all_mean = loss_all_sum / tot
    teacher_student_acc = 100. * teacher_student_correct_sum / tot
    print('{} set: Average loss: {:.4f}, Teacher-Student Accuracy: {}/{} ({:.0f}% )'.format(
        val_test, loss_all_mean, teacher_student_correct_sum, tot, teacher_student_acc))
    return loss_all_mean, teacher_student_acc


def test(args, model, netS, device, test_loader, epoch, val_test, temp):
    model.eval()
    netS.eval()
    loss_all_sum = 0
    tot = 0
    student_correct_sum = 0
    teacher_student_correct_sum = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            tot += data.shape[0]
            data = data*2 - 1
            output_teacher_logits = model(data)
            output_student_logits = netS(data)
            output_teacher_logits_ht = output_teacher_logits / temp
            output_student_logits_ht = output_student_logits / temp
            sm_teacher_ht = F.softmax(output_teacher_logits_ht,dim=1)
            sm_student_ht = F.softmax(output_student_logits_ht,dim=1)
            sm_teacher = F.softmax(output_teacher_logits, dim=1)
            sm_student = F.softmax(output_student_logits, dim=1)
            loss_kd = nn.KLDivLoss(reduction='sum')(F.log_softmax(output_student_logits_ht, dim=1),F.softmax(output_teacher_logits_ht, dim=1))
            pred_class_argmax_teacher = sm_teacher.max(1, keepdim=True)[1]
            loss_ce = F.cross_entropy(output_student_logits, pred_class_argmax_teacher.view(data.shape[0]),reduction='sum')
            loss_all = args.lambda_*temp*temp*loss_kd + (1-args.lambda_)*loss_ce
            loss_all_sum += loss_all
            pred_class_argmax_student = sm_student.max(1, keepdim=True)[1]
            pred_class_argmax_teacher = pred_class_argmax_teacher.view(sm_teacher.shape[0])
            pred_class_argmax_student = pred_class_argmax_student.view(sm_teacher.shape[0])
            student_correct = torch.sum(pred_class_argmax_student==target)
            student_correct_sum = student_correct_sum + (student_correct).cpu().data.numpy()
            teacher_student_correct = torch.sum(pred_class_argmax_student==pred_class_argmax_teacher)
            teacher_student_correct_sum = teacher_student_correct_sum + (teacher_student_correct).cpu().data.numpy()
    loss_all_mean = loss_all_sum / tot
    student_acc = 100. * student_correct_sum / tot
    teacher_student_acc = 100. * teacher_student_correct_sum / tot
    print('{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Teacher-Student Accuracy: {}/{} ({:.0f}% )'.format(
        val_test, loss_all_mean, student_correct_sum, tot, student_acc, teacher_student_correct_sum, tot, teacher_student_acc))
    return loss_all_mean, student_acc, teacher_student_acc

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='CIFAR Classifier training')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--temp', default=20, type=float, help='Temperature for KD')
    parser.add_argument('--lambda_', default=1, type=float, help='Weight of KD Loss during distillation')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    tfm = transforms.Compose([
            transforms.ToTensor()
    ])
    train_dataset = datasets.CIFAR100(
        root='../../../../datasets', train=True,
        download=True, transform=tfm)
    val_dataset = datasets.CIFAR100(
        root='../../../../datasets', train=True,
        download=True, transform=tfm)
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(0.2 * num_train))
    np.random.seed(args.seed)
    np.random.shuffle(indices)
    train_idx, val_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.test_batch_size, sampler=val_sampler,**kwargs
    )   
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../../../../datasets', train=False, download=True, 
                        transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    model = AlexNet().to(device)
    model.eval()
    model.load_state_dict(torch.load("../CIFAR10_data/best_model.pth"))
    netS = AlexNet_half().to(device)
    #netS.load_state_dict(torch.load("./models/best_model.pth"))
    optimizer = optim.Adam(netS.parameters(), lr=args.lr)
    best_val_acc = 0
    cnt = 0
    temp = args.temp
    # Used classes of CIFAR-100
    inc_classes = [70, 47, 49, 37, 86, 53, 16, 94, 54, 25]
    for epoch in range(1, args.epochs + 1):
        train_loss_kd, train_teacher_student_acc = train(args, model, netS, device, train_loader, optimizer, epoch, temp, inc_classes)
        val_loss_kd, val_teacher_student_acc = val(args, model, netS, device, val_loader, epoch, 'Validation', temp, inc_classes)
        test_loss_kd, test_student_acc, test_teacher_student_acc = test(args, model, netS, device, test_loader, epoch, 'Test', temp)
        if val_teacher_student_acc > best_val_acc:
            print("Saving best model...")
            torch.save(netS.state_dict(), "models/best_model_lr1.pth")
            best_val_acc = val_teacher_student_acc
            cnt = 0
            train_st_acc_lr1 = train_teacher_student_acc
            val_st_acc_lr1 = val_teacher_student_acc
            test_st_acc_lr1 = test_teacher_student_acc
            test_acc_lr1 = test_student_acc
        else:
            cnt += 1
        writer.add_scalar("1_Train loss", train_loss_kd, epoch)
        writer.add_scalar("2_Validation loss", val_loss_kd, epoch)
        writer.add_scalar("3_Test loss", test_loss_kd, epoch)
        writer.add_scalar("7_Test accuracy", test_student_acc, epoch)
        writer.add_scalar("4_Train Teacher-Student accuracy", train_teacher_student_acc, epoch)
        writer.add_scalar("5_Validation Teacher-Student accuracy", val_teacher_student_acc, epoch)
        writer.add_scalar("6_Test Teacher-Student accuracy", test_teacher_student_acc, epoch)
        if cnt > 100:
            print('Model has converged with learning rate = {}!'.format(args.lr))
            break
    n_epochs_lr1 = epoch
    optimizer = optim.Adam(netS.parameters(), lr=args.lr*0.1)
    netS.load_state_dict(torch.load("models/best_model_lr1.pth"))
    cnt = 0
    train_st_acc_lr2 = train_st_acc_lr1
    val_st_acc_lr2 = val_st_acc_lr1
    test_st_acc_lr2 = test_st_acc_lr1   
    test_acc_lr2 = test_acc_lr1
    torch.save(netS.state_dict(), "models/best_model_lr2.pth")
    for epoch in range(1, args.epochs + 1):
        train_loss_kd, train_teacher_student_acc = train(args, model, netS, device, train_loader, optimizer, epoch + n_epochs_lr1, temp, inc_classes)
        val_loss_kd, val_teacher_student_acc = val(args, model, netS, device, val_loader, epoch + n_epochs_lr1, 'Validation', temp, inc_classes)
        test_loss_kd, test_student_acc, test_teacher_student_acc = test(args, model, netS, device, test_loader, epoch + n_epochs_lr1, 'Test', temp)
        if val_teacher_student_acc > best_val_acc:
            print("Saving best model...")
            torch.save(netS.state_dict(), "models/best_model_lr2.pth")
            best_val_acc = val_teacher_student_acc
            cnt = 0
            train_st_acc_lr2 = train_teacher_student_acc
            val_st_acc_lr2 = val_teacher_student_acc
            test_st_acc_lr2 = test_teacher_student_acc
            test_acc_lr2 = test_student_acc
        else:
            cnt += 1
        writer.add_scalar("1_Train loss", train_loss_kd, epoch + n_epochs_lr1)
        writer.add_scalar("2_Validation loss", val_loss_kd, epoch + n_epochs_lr1)
        writer.add_scalar("3_Test loss", test_loss_kd, epoch + n_epochs_lr1)
        writer.add_scalar("7_Test accuracy", test_student_acc, epoch + n_epochs_lr1)
        writer.add_scalar("4_Train Teacher-Student accuracy", train_teacher_student_acc, epoch + n_epochs_lr1)
        writer.add_scalar("5_Validation Teacher-Student accuracy", val_teacher_student_acc, epoch + n_epochs_lr1)
        writer.add_scalar("6_Test Teacher-Student accuracy", test_teacher_student_acc, epoch + n_epochs_lr1)
        if cnt > 100:
            print('Model has converged with learning rate = {}!'.format(args.lr*0.1))
            break

    n_epochs_lr2 = epoch
    print('Number of epochs with lr = {} are {} and number of epochs with lr = {} are {}'.format(
        args.lr, n_epochs_lr1, args.lr*0.1, n_epochs_lr2))
    print('Accuracy with lr = {}: Train ST accuracy = {:.2f}%, Validation ST accuracy = {:.2f}%, Test ST accuracy = {:.2f}%, Test accuracy = {:.2f}%'.format(
        args.lr, train_st_acc_lr1, val_st_acc_lr1, test_st_acc_lr1, test_acc_lr1))
    print('Accuracy with lr = {}: Train ST accuracy = {:.2f}%, Validation ST accuracy = {:.2f}%, Test ST accuracy = {:.2f}%, Test accuracy = {:.2f}%'.format(
        args.lr*0.1, train_st_acc_lr2, val_st_acc_lr2, test_st_acc_lr2, test_acc_lr2))
 
    writer.close()

if __name__ == '__main__':
    main()

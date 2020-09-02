# Code to distill the knowledge from a Teacher to Student using data generated by a Generator

from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import numpy as np
from dcgan_model import Generator, Discriminator
from alexnet import AlexNet
from alexnet_half import AlexNet_half

writer = SummaryWriter()

# CUDA_VISIBLE_DEVICES=0 python KD_dfgan.py --dataroot ../../../../datasets --cuda --outf models --manualSeed 108 --niter 5000 --lambda_ 1 --temp 20 --netG ../../train_generator/out_cifar/netG_epoch_199.pth --netC ./best_model.pth

if __name__ == '__main__':    

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=True, help='path to test dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', required=True, help="path to Generator network weights")
    parser.add_argument('--netC', required=True, help="path to Teacher network weights")
    parser.add_argument('--netS', default='', help="path to Student network weights (to continue training)")
    parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--temp', default=10, type=float, help='Temperature for KD')
    parser.add_argument('--lambda_', default=1, type=float, help='Weight of KD Loss during distillation')
    parser.add_argument('--nBatches', default=256, type=float, help='Number of Batches')

    opt = parser.parse_args()
    print(opt)

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    nc=3

    transform=transforms.Compose([
                        transforms.ToTensor(),
                    ])

    test_loader = torch.utils.data.DataLoader(
        dset.CIFAR10(opt.dataroot, train=False, download=True, transform=transform),
            batch_size=opt.batchSize, shuffle=False)

    device = torch.device("cuda:0" if opt.cuda else "cpu")
    ngpu = int(opt.ngpu)
    nz = int(opt.nz)

    netG = Generator(ngpu).to(device)
    netG.load_state_dict(torch.load(opt.netG))
    print(netG)
    netG.eval()

    netC = AlexNet().to(device)
    netC.load_state_dict(torch.load(opt.netC))
    print(netC)
    netC.eval()

    netS = AlexNet_half().to(device)
    if opt.netS != '':
        netS.load_state_dict(torch.load(opt.netS))
    print(netS)
    temp = opt.temp
    batch_size = int(opt.batchSize)
    n_batches = int(opt.nBatches)
    # setup optimizer
    threshold = []
    best_val_acc = 0
    cnt = 0
    n_epochs_lr1 = 0
    for lr_cnt in range(2):
        if lr_cnt == 0:
            lrate = opt.lr
        else:
            lrate = opt.lr * 0.1
            netS.load_state_dict(torch.load('models/best_model_lr1.pth'))
            train_st_acc_lr2 = train_st_acc_lr1
            val_st_acc_lr2 = val_st_acc_lr1
            test_st_acc_lr2 = test_st_acc_lr1
            test_acc_lr2 = test_acc_lr1
            torch.save(netS.state_dict(), "models/best_model_lr2.pth")
        optimizerS = optim.Adam(netS.parameters(), lr=lrate, betas=(opt.beta1, 0.999))
        for epoch in range(1, opt.niter+1):
            loss_kd_sum = 0
            loss_ce_sum = 0
            loss_all_sum = 0
            teacher_student_correct_sum = 0
            netS.train()
            for i in range(n_batches):
                optimizerS.zero_grad()
                noise_rand = torch.randn(batch_size, nz, 1, 1, device=device)
                fake_train = netG(noise_rand)
                fake_train_class = netC(fake_train)
                fake_student_class = netS(fake_train)
                fake_train_class_ht = fake_train_class/temp
                fake_student_class_ht = fake_student_class/temp
                sm_teacher_ht = F.softmax(fake_train_class_ht, dim=1)
                sm_student_ht = F.softmax(fake_student_class_ht, dim=1)
                sm_teacher = F.softmax(fake_train_class, dim=1)
                sm_student = F.softmax(fake_student_class, dim=1)
                pred_class_argmax_teacher = sm_teacher.max(1, keepdim=True)[1]
                loss_kd = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(fake_student_class_ht, dim=1),F.softmax(fake_train_class_ht, dim=1))
                loss_ce = F.cross_entropy(fake_student_class, pred_class_argmax_teacher.view(batch_size))
                loss_all = opt.lambda_*temp*temp*loss_kd + (1-opt.lambda_)*loss_ce
                loss_kd_sum = loss_kd_sum + loss_kd
                loss_ce_sum = loss_ce_sum + loss_ce
                loss_all_sum = loss_all_sum + loss_all
                loss_all.backward()
                optimizerS.step()
                pred_class_argmax_student = sm_student.max(1, keepdim=True)[1]
                pred_class_argmax_teacher = pred_class_argmax_teacher.view(sm_teacher.shape[0])
                pred_class_argmax_student = pred_class_argmax_student.view(sm_teacher.shape[0])
                teacher_student_correct = torch.sum(pred_class_argmax_student==pred_class_argmax_teacher)
                teacher_student_correct_sum = teacher_student_correct_sum + (teacher_student_correct).cpu().data.numpy()
            # do checkpointing
            torch.save(netS.state_dict(), '%s/netS_epoch_%d.pth' % (opt.outf, epoch + n_epochs_lr1))
            loss_kd_val_sum = 0
            loss_ce_val_sum = 0
            loss_all_val_sum = 0
            teacher_student_correct_val_sum = 0
            netS.eval()
            with torch.no_grad():
                for i in range(int(np.floor(n_batches/4))):
                    noise_rand = torch.randn(batch_size, nz, 1, 1, device=device)
                    fake_train = netG(noise_rand)
                    fake_train_class = netC(fake_train)
                    fake_student_class = netS(fake_train)
                    fake_train_class_ht = fake_train_class/temp
                    fake_student_class_ht = fake_student_class/temp
                    sm_teacher_ht = F.softmax(fake_train_class_ht, dim=1)
                    sm_student_ht = F.softmax(fake_student_class_ht, dim=1)
                    sm_teacher = F.softmax(fake_train_class, dim=1)
                    sm_student = F.softmax(fake_student_class, dim=1)
                    pred_class_argmax_teacher = sm_teacher.max(1, keepdim=True)[1]
                    loss_kd = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(fake_student_class_ht, dim=1),F.softmax(fake_train_class_ht, dim=1))
                    loss_ce = F.cross_entropy(fake_student_class, pred_class_argmax_teacher.view(batch_size))
                    loss_all = opt.lambda_*temp*temp*loss_kd + (1-opt.lambda_)*loss_ce
                    loss_kd_val_sum = loss_kd_val_sum + loss_kd
                    loss_ce_val_sum = loss_ce_val_sum + loss_ce
                    loss_all_val_sum = loss_all_val_sum + loss_all
                    pred_class_argmax_student = sm_student.max(1, keepdim=True)[1]
                    pred_class_argmax_teacher = pred_class_argmax_teacher.view(sm_teacher.shape[0])
                    pred_class_argmax_student = pred_class_argmax_student.view(sm_teacher.shape[0])
                    teacher_student_correct = torch.sum(pred_class_argmax_student==pred_class_argmax_teacher)
                    teacher_student_correct_val_sum = teacher_student_correct_val_sum + (teacher_student_correct).cpu().data.numpy()
                teacher_acc_sum = 0.0
                student_acc_sum = 0.0
                teacher_student_correct_test_sum = 0.0
                num = 0.0
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    data = data*2 - 1
                    test_class_teacher = netC(data)
                    test_class_student = netS(data)
                    sm_teacher_test = F.softmax(test_class_teacher, dim=1)
                    sm_student_test = F.softmax(test_class_student, dim=1)
                    pred_class_argmax_teacher_test = sm_teacher_test.max(1, keepdim=True)[1]
                    pred_class_argmax_student_test = sm_student_test.max(1, keepdim=True)[1]
                    pred_class_argmax_teacher_test = pred_class_argmax_teacher_test.view(target.shape[0])
                    pred_class_argmax_student_test = pred_class_argmax_student_test.view(target.shape[0])
                    teacher_acc = torch.sum(pred_class_argmax_teacher_test==target)
                    student_acc = torch.sum(pred_class_argmax_student_test==target)
                    teacher_acc_sum = teacher_acc_sum + teacher_acc
                    student_acc_sum = student_acc_sum + student_acc
                    num = num + target.shape[0]
                    teacher_student_correct = torch.sum(pred_class_argmax_student_test==pred_class_argmax_teacher_test)
                    teacher_student_correct_test_sum = teacher_student_correct_test_sum + (teacher_student_correct).cpu().data.numpy()
                teacher_acc_mean = float(teacher_acc_sum) / float(num)
                student_acc_mean = float(student_acc_sum) / float(num)
                teacher_student_correct_test_mean = float(teacher_student_correct_test_sum) / float(num)
            val_student_acc = teacher_student_correct_val_sum / (float(np.floor(n_batches/4))*batch_size)
            train_student_acc = teacher_student_correct_sum/ float(n_batches*batch_size)
            if val_student_acc > best_val_acc:
                print("Saving best model...")
                if lr_cnt ==0 :
                    torch.save(netS.state_dict(), "models/best_model_lr1.pth")
                    train_st_acc_lr1 = train_student_acc
                    val_st_acc_lr1 = val_student_acc
                    test_st_acc_lr1 = teacher_student_correct_test_mean
                    test_acc_lr1 = student_acc_mean
                else:
                    torch.save(netS.state_dict(), "models/best_model_lr2.pth")
                    train_st_acc_lr2 = train_student_acc
                    val_st_acc_lr2 = val_student_acc
                    test_st_acc_lr2 = teacher_student_correct_test_mean
                    test_acc_lr2 = student_acc_mean
                best_val_acc = val_student_acc
                cnt = 0
                
            else:
                cnt += 1
            print("Epoch",epoch + n_epochs_lr1,"/",opt.niter)
            print("Teacher accuracy=",round(teacher_acc_mean*100,2),"%, Student accuracy=",round(student_acc_mean*100,2),"%")
            writer.add_scalar("KD loss train", loss_kd_sum/ n_batches, epoch + n_epochs_lr1)
            writer.add_scalar("KD loss val", loss_kd_val_sum/ float(np.floor(n_batches/4)), epoch + n_epochs_lr1)
            writer.add_scalar("CE loss train", loss_ce_sum/ n_batches, epoch + n_epochs_lr1)
            writer.add_scalar("CE loss val", loss_ce_val_sum/ float(np.floor(n_batches/4)), epoch + n_epochs_lr1)
            writer.add_scalar("Total loss train", loss_all_sum/ n_batches, epoch + n_epochs_lr1)
            writer.add_scalar("Total loss val", loss_all_val_sum/ float(np.floor(n_batches/4)), epoch + n_epochs_lr1)
            writer.add_scalar("Student test accuracy", student_acc_mean, epoch + n_epochs_lr1)
            writer.add_scalar("Teacher-Student train accuracy", train_student_acc, epoch + n_epochs_lr1)
            writer.add_scalar("Teacher-Student val accuracy", val_student_acc, epoch + n_epochs_lr1)
            writer.add_scalar("Teacher-Student test accuracy", teacher_student_correct_test_mean, epoch + n_epochs_lr1)
            writer.export_scalars_to_json("./all_scalars.json")
            if cnt > 100:
                print('Model has converged with learning rate = {}!'.format(lrate))
                cnt = 0 
                break
        if lr_cnt == 0:
            n_epochs_lr1 = epoch
        else:
            n_epochs_lr2 = epoch 
    print('Number of epochs with lr = {} are {} and number of epochs with lr = {} are {}'.format(
        opt.lr, n_epochs_lr1, opt.lr*0.1, n_epochs_lr2))
    print('Accuracy with lr = {}: Train ST accuracy = {:.2f}%, Validation ST accuracy = {:.2f}%, Test ST accuracy = {:.2f}%, Test accuracy = {:.2f}%'.format(
        opt.lr, train_st_acc_lr1*100, val_st_acc_lr1*100, test_st_acc_lr1*100, test_acc_lr1*100))
    print('Accuracy with lr = {}: Train ST accuracy = {:.2f}%, Validation ST accuracy = {:.2f}%, Test ST accuracy = {:.2f}% Test accuracy = {:.2f}%'.format(
        opt.lr*0.1, train_st_acc_lr2*100, val_st_acc_lr2*100, test_st_acc_lr2*100, test_acc_lr2*100))
 
writer.close()


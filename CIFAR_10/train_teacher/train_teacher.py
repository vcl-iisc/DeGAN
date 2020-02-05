# Training a Classifier on CIFAR-10 using AlexNet architecture
from __future__ import print_function
import argparse
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from alexnet import AlexNet
from torch.utils.data.sampler import SubsetRandomSampler
from tensorboardX import SummaryWriter
import numpy as np

# CUDA_VISIBLE_DEVICES=0 python train_teacher.py --batch-size 64 --test-batch-size 1000 --epochs 1000 --lr 0.001 --seed 108 --log-interval 10
writer = SummaryWriter()
if not os.path.exists("models"):
    os.makedirs("models")

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    tot = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        tot += data.shape[0]
        optimizer.zero_grad()
        data = data*2 - 1
        logits = model(data)
        output = F.softmax(logits,dim=1)
        ce_loss = nn.CrossEntropyLoss()
        loss = ce_loss(logits, target)
        loss.backward()
        ce_loss_redn = nn.CrossEntropyLoss(reduction = 'sum')
        train_loss += ce_loss_redn(logits, target).item()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, int((batch_idx+1) * len(data)), int(len(train_loader.dataset)*0.8),
                100. * (batch_idx+1) / len(train_loader), loss.item()))
    train_loss /= tot
    train_acc = 100. * correct / tot
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, tot,train_acc))
    torch.save(model.state_dict(), "models/"+str(epoch)+".pth")
    return train_loss, train_acc

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data*2 - 1
            logits = model(data)
            output = F.softmax(logits,dim=1)
            ce_loss_redn = nn.CrossEntropyLoss(reduction = 'sum')
            test_loss += ce_loss_redn(logits, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset), test_acc))
    return test_loss, test_acc

def val(args, model, device, val_loader):
    model.eval()
    val_loss = 0
    correct = 0
    tot = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            data = data*2 - 1
            logits = model(data)
            output = F.softmax(logits,dim=1)
            ce_loss_redn = nn.CrossEntropyLoss(reduction = 'sum')
            val_loss += ce_loss_redn(logits, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            tot += data.shape[0]
    val_loss /= tot
    val_acc = 100. * correct / tot
    print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        val_loss, correct, tot, val_acc))
    return val_loss, val_acc

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
    train_dataset = datasets.CIFAR10(
        root='../../../datasets', train=True,
        download=True, transform=tfm)
    val_dataset = datasets.CIFAR10(
        root='../../../datasets', train=True,
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
        val_dataset, batch_size=args.batch_size, sampler=val_sampler,**kwargs
    )   
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../../../datasets', train=False, download=True, 
                        transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    model = AlexNet().to(device)
    #model.load_state_dict(torch.load("./best_model.pth"))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_val_acc = 0
    cnt = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(args, model, device, train_loader, optimizer, epoch)
        val_loss, val_acc = val(args, model, device, val_loader)
        test_loss, test_acc = test(args, model, device, test_loader)
        if val_acc > best_val_acc:
            print("Saving best model...")
            torch.save(model.state_dict(), "models/best_model_lr1.pth")
            best_val_acc = val_acc
            cnt = 0
            train_acc_lr1 = train_acc
            val_acc_lr1 = val_acc
            test_acc_lr1 = test_acc
        else:
            cnt += 1
        writer.add_scalar("1_Train loss", train_loss, epoch)
        writer.add_scalar("2_Validation loss", val_loss, epoch)
        writer.add_scalar("3_Test loss", test_loss, epoch)
        writer.add_scalar("4_Train accuracy", train_acc, epoch)
        writer.add_scalar("5_Validation accuracy", val_acc, epoch)
        writer.add_scalar("6_Test accuracy", test_acc, epoch)
        writer.export_scalars_to_json("./all_scalars.json")
        if cnt > 100:
            print('Model has converged with learning rate = {}!'.format(args.lr))
            break
    n_epochs_lr1 = epoch
    optimizer = optim.Adam(model.parameters(), lr=args.lr*0.1)
    model.load_state_dict(torch.load("models/best_model_lr1.pth"))
    cnt = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(args, model, device, train_loader, optimizer, epoch + n_epochs_lr1)
        val_loss, val_acc = val(args, model, device, val_loader)
        test_loss, test_acc = test(args, model, device, test_loader)
        if val_acc > best_val_acc:
            print("Saving best model...")
            torch.save(model.state_dict(), "models/best_model_lr2.pth")
            best_val_acc = val_acc
            cnt = 0
            train_acc_lr2 = train_acc
            val_acc_lr2 = val_acc
            test_acc_lr2 = test_acc

        else:
            cnt += 1
        writer.add_scalar("1_Train loss", train_loss, epoch + n_epochs_lr1)
        writer.add_scalar("2_Validation loss", val_loss, epoch + n_epochs_lr1)
        writer.add_scalar("3_Test loss", test_loss, epoch + n_epochs_lr1)
        writer.add_scalar("4_Train accuracy", train_acc, epoch + n_epochs_lr1)
        writer.add_scalar("5_Validation accuracy", val_acc, epoch + n_epochs_lr1)
        writer.add_scalar("6_Test accuracy", test_acc, epoch + n_epochs_lr1)
        writer.export_scalars_to_json("./all_scalars.json")
        if cnt > 100:
            print('Model has converged with learning rate = {}!'.format(args.lr*0.1))
            break
    n_epochs_lr2 = epoch 
    print('Number of epochs with lr = {} are {} and number of epochs with lr = {} are {}'.format(
        args.lr, n_epochs_lr1, args.lr*0.1, n_epochs_lr2))
    print('Accuracy with lr = {}: Train accuracy = {:.2f}%, Validation accuracy = {:.2f}%, Test accuracy = {:.2f}%'.format(
        args.lr, train_acc_lr1, val_acc_lr1, test_acc_lr1))
    print('Accuracy with lr = {}: Train accuracy = {:.2f}%, Validation accuracy = {:.2f}%, Test accuracy = {:.2f}%'.format(
        args.lr*0.1, train_acc_lr2, val_acc_lr2, test_acc_lr2)) 
    
    writer.close()
    
if __name__ == '__main__':
    main()



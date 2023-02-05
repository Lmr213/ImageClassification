import torch
import visdom
from torch import nn, optim

from torch.utils.data import Dataset, DataLoader
from preData import Pokemon
from resnet import ResNet

batch_size = 32
lr = 1e-3
epochs = 10

# set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1234)

train_ds = Pokemon('pokeman', 224, mode='train')
valid_ds = Pokemon('pokeman', 224, mode='val')
test_ds = Pokemon('pokeman', 224, mode='test')

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_ds, batch_size=batch_size, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=2)

vis = visdom.Visdom()


# test (不需要计算梯度的部分，不用构建计算图)

def evalute(model, loader):  # 测试函数对validation和test一样的功能
    model.eval()  # test mode（声明模式因为在train和test时候模型的计算可能不同，eg:用不用dropout）
    correct = 0
    total = len(loader.dataset)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():  # 不需要反向传播计算梯度
            output = model(x)
            pred = output.argmax(dim=1)  # 取dim维度上最大值的索引
        correct += torch.eq(pred, y).sum().float().item()  # [b] vs [b] => scalar tensor

    return correct / total


def main():

    model = ResNet(5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss().to(device)

    best_acc, best_epoch = 0, 0
    vis.line([0], [-1], win='loss', opts=dict(title='loss'))
    vis.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))
    global_step = 0
    for epoch in range(epochs):

        for step, (x, y) in enumerate(train_loader):

            # x: [b, 3, 224, 224], y:[b]
            x, y = x.to(device), y.to(device)

            # y_h: [b, 10], y: [b]
            # loss: tensor scalar
            model.train()  # train mode
            output = model(x)
            loss = loss_function(output, y)

            # backprop
            optimizer.zero_grad()  # 每次要先梯度清零（因为每次backprop会把新的梯度累加到之前的梯度上面）
            loss.backward()
            optimizer.step()

            vis.line([loss.item()], [global_step], win='loss', update='append')  # 把当前step的loss保存下来存到曲线的末尾
            global_step += 1

        if epoch % 2 == 0:   # 每两次epoch做一次validation

            val_acc = evalute(model, valid_loader)
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                # save the current mode，下次继续用
                torch.save(model.state_dict(), 'best.mdl')

                vis.line([val_acc], [global_step], win='val_acc', update='append')

    print('best acc:', best_acc, 'best epoch:', best_epoch)

    model.load_state_dict((torch.load('best.mdl')))  # test时候加载最准确的模型的状态

    test_acc = evalute(model, test_loader)
    print('test acc:', test_acc)



if __name__ == '__main__':
    main()

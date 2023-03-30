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


# test 

def evalute(model, loader):  
    model.eval()  # test mode
    correct = 0
    total = len(loader.dataset)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad(): 
            output = model(x)
            pred = output.argmax(dim=1)  
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
            optimizer.zero_grad()  
            loss.backward()
            optimizer.step()

            vis.line([loss.item()], [global_step], win='loss', update='append')  
            global_step += 1

        if epoch % 2 == 0:   # each two epoch do validation

            val_acc = evalute(model, valid_loader)
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                # save the current mode
                torch.save(model.state_dict(), 'best.mdl')

                vis.line([val_acc], [global_step], win='val_acc', update='append')

    print('best acc:', best_acc, 'best epoch:', best_epoch)

    model.load_state_dict((torch.load('best.mdl')))

    test_acc = evalute(model, test_loader)
    print('test acc:', test_acc)



if __name__ == '__main__':
    main()

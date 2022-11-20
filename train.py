import torch
import torch.nn.functional as F
import torch.optim as optim

from alexnet import AlexNet
import datasets
from loguru import logger


def train():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataloader = datasets.train_dataloader
    model = AlexNet()
    model.to(DEVICE)

    optimizer = optim.SGD(model.parameters(),
                          lr=0.01,
                          momentum=0.9,
                          weight_decay=0.0005)

    model.train()
    for epoch_index in range(10):
        for batch_index, (data, label) in enumerate(train_dataloader):
            data, label = data.to(DEVICE), label.to(DEVICE)

            optimizer.zero_grad()

            output = model(data)

            loss = F.cross_entropy(output, label)

            loss.backward()

            optimizer.step()

            if batch_index % 100 == 0:
                logger.info(
                    'train epoch:{} batch:{}\ttrain loss:{:.6f}'.format(
                        epoch_index, batch_index, loss.item()))

    torch.save(model, './checkpoints/alexnet.pth')
    logger.info('成功保存模型在:{}'.format('./checkpoints/alexnet.pth'))


if __name__ == '__main__':
    train()
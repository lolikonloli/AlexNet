import torch
import torch.nn.functional as F

import datasets
from loguru import logger


def test():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataloader = datasets.test_dataloader

    model = torch.load('./checkpoints/alexnet.pth')
    model.eval()

    correct = 0.0
    test_loss = 0.0

    #不需要梯度的记录
    with torch.no_grad():
        for batch_index, (data, label) in enumerate(test_dataloader):
            data, label = data.to(DEVICE), label.to(DEVICE)

            output = model(data)

            loss = F.cross_entropy(output, label).item()
            if batch_index % 100 == 0:
                logger.info('batch:{}, loss:{}'.format(batch_index, loss))

            test_loss += loss

            prediction = output.argmax(dim=1)

            correct += prediction.eq(label.view_as(prediction)).sum().item()

        test_loss /= len(test_dataloader.dataset)
        accuracy = 100 * correct / len(test_dataloader.dataset)

        logger.info('average_loss:{:.4f}, accuracy:{:3f}\n'.format(
            test_loss, accuracy))


if __name__ == '__main__':
    test()

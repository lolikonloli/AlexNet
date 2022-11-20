import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import datasets
from loguru import logger


def test():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataloader = datasets.test_dataloader

    model = torch.load('./checkpoints/alexnet.pth')
    model.eval()

    #不需要梯度的记录
    with torch.no_grad():
        for batch_index, (data, label) in enumerate(test_dataloader):
            data, label = data.to(DEVICE), label.to(DEVICE)

            output = model(data)

            loss = F.cross_entropy(output, label).item()
            logger.info('batch:{}, loss:{}'.format(batch_index, loss))

            prediction = output.argmax(dim=1)

            logger.info('图片为：{} 预测为:{} '.format(label[0], prediction[0]))
            plt.imshow(data.to('cpu').squeeze().numpy())
            plt.show()


if __name__ == '__main__':
    test()

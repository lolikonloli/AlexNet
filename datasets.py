from torch.utils.data.dataloader import DataLoader
import torchvision
import torchvision.transforms as transforms

train_datasets = torchvision.datasets.MNIST('./data',
                                            download=True,
                                            train=True,
                                            transform=transforms.ToTensor())

train_dataloader = DataLoader(train_datasets, batch_size=16, shuffle=True)

test_datasets = torchvision.datasets.MNIST('./data',
                                            download=True,
                                            train=False,
                                            transform=transforms.ToTensor())

test_dataloader = DataLoader(train_datasets, batch_size=1, shuffle=True)

if __name__ == '__main__':
    from loguru import logger
    import matplotlib.pyplot as plt

    for img, label in train_dataloader:
        logger.info('img:{}'.format(type(img)))
        logger.info('label:{}'.format(type(label)))

        logger.info('label:{}'.format(label[0]))
        plt.imshow(img[0].squeeze().numpy(), cmap='Greys_r')
        plt.show()

        # import pdb
        # pdb.set_trace()

        break

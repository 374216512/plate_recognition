import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import CRNN
from torchsummary import summary
from data import CustomDataset
from tqdm import tqdm
import utils


def train(train_path, test_path):
    alphabet = "京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"
    imgh = 32  # the height of the input image to network
    nc = 3
    nclass = len(alphabet) + 1
    nh = 256  # size of the lstm hidden state
    # word2vec
    converter = utils.strLabelConverter(alphabet)

    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CRNN(imgh, nc, nclass, nh).to(device)

    # 加载数据
    batch_size = 32
    train_set = CustomDataset(train_path)
    test_set = CustomDataset(test_path)

    # 定义超参数
    epochs = 30
    criterion = nn.CTCLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=1e-3)

    # 开始训练
    for epoch in range(epochs):
        train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        train_iter = tqdm(train_dataloader, total=len(train_dataloader))
        test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
        test_iter = tqdm(test_dataloader, total=len(test_dataloader))

        for index, (x_batch, y_batch) in enumerate(train_iter):
            if index == len(train_iter) - 1:
                continue
            x_batch = x_batch.to(device)
            text, length = converter.encode(y_batch)  # 编码

            # Forward
            output = model(x_batch)
            output_shape = Variable(torch.IntTensor([output.size(0)] * batch_size))
            loss = criterion(output, text, output_shape, length) / batch_size

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_iter.desc = f'Epoch: {epoch}'
            train_iter.postfix = f'Loss: {loss.item()}'

        torch.save(model, 'output/model.pt')

if __name__ == '__main__':
    train_path = r'D:\code\DLProject\datasets\CCPD2020-voc\train\plates'
    test_path = r'D:\code\DLProject\datasets\CCPD2020-voc\test\plates'
    train(train_path, test_path)

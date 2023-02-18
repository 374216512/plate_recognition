import torch
from torch.autograd import Variable
import utils
from PIL import Image
from PIL import Image

from model import CRNN
from torchvision import transforms

model_path = './output/model.pt'
img_path = '皖AD00657_02569_000.jpg'
alphabet = "京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(model_path)

converter = utils.strLabelConverter(alphabet)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 168))
])

img = Image.open(img_path)
img = transform(img)
img = torch.unsqueeze(img, 0).to(device)

model.eval()
preds = model(img)

_, preds = preds.max(2)
preds = preds.transpose(1, 0).contiguous().view(-1)

preds_size = Variable(torch.IntTensor([preds.size(0)]))
raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
print('%-20s => %-20s' % (raw_pred, sim_pred))

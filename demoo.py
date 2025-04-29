import argparse
import scipy.io
import torch
import numpy as np
import os
from torchvision import datasets
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import shutil


#######################################################################
# 自定义数据集类（直接加载无类别的图片）
class ImageListDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_paths = [
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, img_path  # 返回图片和路径（无类别标签）


#######################################################################
# 参数解析
parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--query_index', default=0, type=int, help='查询图片的索引')
parser.add_argument('--test_dir', default='./data/test', type=str, help='测试数据目录')
opts = parser.parse_args()

# 数据集名称
gallery_name = 'workshop_gallery_satellite'
query_name = 'workshop_query_street'
data_dir = opts.test_dir

# 检查数据目录是否存在
gallery_dir = os.path.join(data_dir, gallery_name)
query_dir = os.path.join(data_dir, query_name)
if not os.path.exists(gallery_dir):
    raise FileNotFoundError(f"Gallery目录不存在: {gallery_dir}")
if not os.path.exists(query_dir):
    raise FileNotFoundError(f"Query目录不存在: {query_dir}")

# 加载数据集（使用自定义Dataset）
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
image_datasets = {
    'gallery': ImageListDataset(gallery_dir, transform),
    'query': ImageListDataset(query_dir, transform)
}


#####################################################################
# 显示图片函数
def imshow(path, title=None):
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.1)


######################################################################
# 加载预提取特征
result = scipy.io.loadmat('pytorch_result.mat')
query_feature = torch.FloatTensor(result['query_f'])
query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_label = result['gallery_label'][0]

# 多查询处理（若有）
multi = os.path.isfile('multi_query.mat')
if multi:
    m_result = scipy.io.loadmat('multi_query.mat')
    mquery_feature = torch.FloatTensor(m_result['mquery_f'])
    mquery_label = m_result['mquery_label'][0]
    mquery_feature = mquery_feature.cuda()

query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()


#######################################################################
# 排序函数
def sort_img(qf, ql, gf, gl):
    query = qf.view(-1, 1)
    score = torch.mm(gf, query).squeeze(1).cpu().numpy()
    index = np.argsort(score)[::-1]  # 从高到低排序
    junk_index = np.argwhere(gl == -1)
    mask = np.in1d(index, junk_index, invert=True)
    return index[mask]


i = opts.query_index
index = sort_img(query_feature[i], query_label[i], gallery_feature, gallery_label)

########################################################################
# 可视化结果
query_path = image_datasets['query'].img_paths[i]
print(f"查询图片路径: {query_path}")
print('相似度最高的10张图片:')

save_folder = f'image_show/{opts.query_index:02d}'
os.makedirs(save_folder, exist_ok=True)
shutil.copy(query_path, f'{save_folder}/query.jpg')

try:
    fig = plt.figure(figsize=(16, 4))
    ax = plt.subplot(1, 11, 1)
    ax.axis('off')
    imshow(query_path, 'Query Image')

    for i in range(10):
        ax = plt.subplot(1, 11, i + 2)
        ax.axis('off')
        img_path = image_datasets['gallery'].img_paths[index[i]]
        label = gallery_label[index[i]]
        imshow(img_path)
        shutil.copy(img_path, f'{save_folder}/rank_{i + 1}.jpg')
        ax.set_title(f'{i + 1}', color='green' if label == query_label[i] else 'red')
        print(f"Rank {i + 1}: {img_path}")

    plt.savefig("ranking_result.png")
    plt.show()
except Exception as e:
    print(f"可视化失败（需GUI支持）: {e}")
    for i in range(10):
        print(f"Rank {i + 1}: {image_datasets['gallery'].img_paths[index[i]]}")
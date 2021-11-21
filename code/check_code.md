```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import os
from tqdm.auto import tqdm
from glob import glob
import cv2
import numpy as np
import pandas as pd
import PIL 
import urllib
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from random import uniform
# from imgaug import augmenters as iaa

%config InlineBackend.figure_format = 'retina'
%matplotlib inline
```


```python
import torch.utils.data as td
import torchvision as tv
from PIL import Image
import matplotlib.pyplot as plt
import time
```


```python
import pydicom as dcm
```


```python
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    
device
```




    'cuda'



전처리


*   https://github.com/tuvovan/Unet-with-EfficientnetB7-Backbone/blob/master/Body%20Morphometry.ipynb


모델


*   https://github.com/IanTaehoonYoo/semantic-segmentation-pytorch





```python
# !pip install git+https://github.com/albumentations-team/albumentations
```


```python
# !pip install --user albumentations==1.1.0
```


```python
import albumentations as A
import albumentations.pytorch

aug = A.Compose([
                            # A.HorizontalFlip(p=0.5,always_apply=False),
                            A.OneOf([
                                                                    # A.Random/Contrast(always_apply=False,p=0.2,limit=[-0.1,0.1]),
                                                                    # A.RandomGamma(always_apply=False,p=0.2,gamma_limit=[80,120]),
                                                                    # A.RandomBrightness(always_apply=False,p=0.2,limit=[-0.1,0.1])
                                                                  ], p=0.3),
                            A.OneOf([
                                                                    A.ElasticTransform(always_apply=False,p=0.2,alpha=10,sigma=6.0,alpha_affine=1.5999999999999996,interpolation=1,border_mode=4,approximate=False),
                                                                    A.GridDistortion(always_apply=False,p=0.2,num_steps=1,distort_limit=[-0.1,0.1],interpolation=1,border_mode=4),
                                                                    #albumentations.augmentations.transforms.OpticalDistortion(always_apply=False,p=0.4,distort_limit=[-2,2],shift_limit=[-0.5,0.5],interpolation=1,border_mode=4),                 
                                                                  ], p=0.3),
                            #albumentations.augmentations.transforms.Cutout(always_apply=False,p=0.5,num_holes=8,max_h_size=50,max_w_size=50)
                          #A.ShiftScaleRotate(always_apply=False,p=0.5,shift_limit=[-0.0625,0.0625],scale_limit=[-0.09999999999999998,0.12000000000000009],rotate_limit=[-25,25],interpolation=1,border_mode=4,value=None,mask_value=None),
                          # albumentations.augmentations.transforms.Resize(always_apply=True,p=1,height=512,width=512,interpolation=1)

                    ])
```


```python
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import pydicom

def transform_to_hu(medical_image, image):
    hu_image = image * medical_image.RescaleSlope + medical_image.RescaleIntercept
    hu_image[hu_image < -1024] = -1024
    return hu_image

def window_image(image, window_center, window_width):
    window_image = image.copy()
    image_min = window_center - (window_width / 2)
    image_max = window_center + (window_width / 2)
    window_image[window_image < image_min] = image_min
    window_image[window_image > image_max] = image_max
    return window_image

def resize_normalize(image):
    image = np.array(image, dtype=np.float64)
    image -= np.min(image)
    image /= np.max(image)
    return image

def read_dicom(image_medical, window_widht, window_level):
    image_data = image_medical.pixel_array

    image_hu = transform_to_hu(image_medical, image_data)
    image_window = window_image(image_hu.copy(), window_level, window_widht)
    image_window_norm = resize_normalize(image_window)
#     image_window_norm = image_window

    image_window_norm = np.expand_dims(image_window_norm, axis=2)   # (512, 512, 1)
    image_ths = np.concatenate([image_window_norm, image_window_norm, image_window_norm], axis=2)   # (512, 512, 3)
    #print(image_window_norm.shape)
    return image_ths

def to_binary(img, lower, upper):
    return (lower <= img) & (img <= upper)
```


```python
def mask_binarization(mask, threshold=None):
    if threshold is None:
        threshold = 0.5

    if isinstance(mask, np.ndarray):
        mask_binarized = (mask > threshold).astype(np.uint8)
    
    elif isinstance(mask, torch.Tensor):
        zeros = torch.zeros_like(mask)
        ones = torch.ones_like(mask)
        
        mask_binarized = torch.where(mask > threshold, ones, zeros)
    
    return mask_binarized

def augment_imgs_and_masks(imgs, masks, rot_factor, scale_factor, trans_factor, flip):
    rot_factor = uniform(-rot_factor, rot_factor)
    ran_alp = uniform(10,100)
    scale_factor = uniform(1-scale_factor, 1+scale_factor)
    trans_factor = [int(imgs.shape[1]*uniform(-trans_factor, trans_factor)),
                    int(imgs.shape[2]*uniform(-trans_factor, trans_factor))]

    seq = iaa.Sequential([
            iaa.Affine(
                translate_px={"x": trans_factor[0], "y": trans_factor[1]},
                scale=(scale_factor, scale_factor),
                rotate=rot_factor
            ),
            #iaa.ElasticTransformation(alpha=ran_alp,sigma=5.0)
        
        ])

    seq_det = seq.to_deterministic()

    imgs = seq_det.augment_images(imgs)
    masks = seq_det.augment_images(masks)

    if flip and uniform(0, 1) > 0.5:
        imgs = np.flip(imgs, 2).copy()
        masks = np.flip(masks, 2).copy()
    
    masks = mask_binarization(masks).astype(np.float32)
    return imgs, masks
```


```python
# Data augmentation
rot_factor = 0
scale_factor = 0.0
flip = False
trans_factor = 0.0
```


```python
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, x_dir, y_dir,augmentation=False):
        super().__init__()
        
        self.augmentation = augmentation
        self.x_img = x_dir
        self.y_img = y_dir   

    def __len__(self):
        return len(self.x_img)

    def __getitem__(self, idx):
        x_img = self.x_img[idx]
        y_img = self.y_img[idx]
        # print(x_img)
        # print(y_img)
        # Read an image with OpenCV
        if x_img[-1]=='m' or y_img[-1]=='g':         
            x_img = dcm.read_file(x_img)
            x_img=read_dicom(x_img,400,0)
            x_img=np.transpose(x_img,(2,0,1))
            x_img=x_img.astype(np.float32)
            y_img =  imread(y_img)
            y_img = resize(y_img, (512, 512))*255
            color_im = np.zeros([512, 512, 2])
            for i in range(1,3):
                encode_ = to_binary(y_img, i*1.0, i*1.0)
                color_im[:, :, i-1] = encode_
            color_im = np.transpose(color_im,(2,0,1))
#             y_img = resize(y_img, (512, 512))*255
        else:
            x_img = np.load(x_img)
            x_img=resize_normalize(x_img)
            y_img = np.load(y_img)
            # print(x_img)
            y_img = resize(y_img, (512, 512))
            color_im = np.zeros([512, 512, 2])
            for i in range(1,3):
                encode_ = to_binary(y_img, i*1.0, i*1.0)
                color_im[:, :, i-1] = encode_
            color_im = np.transpose(color_im,(2,0,1))
            image_window_norm = np.expand_dims(x_img, axis=2)   # (512, 512, 1)
            x_img = np.concatenate([image_window_norm, image_window_norm, image_window_norm], axis=2)   # (512, 512, 3)
            x_img=np.transpose(x_img,(2,0,1))
            x_img=x_img.astype(np.float32)
  

       
        # x_img = dcm.read_file(x_img)
        # y_img =  imread(y_img)

        # x_img=read_dicom(x_img,400,50)
        # x_img=np.transpose(x_img,(2,0,1))
        # x_img=x_img.astype(np.float32)

#         y_img = resize(y_img, (512, 512))*255
#         color_im = np.zeros([512, 512, 2])
#         for i in range(1,3):
#             encode_ = to_binary(y_img, i*1.0, i*1.0)
#             color_im[:, :, i-1] = encode_
#         color_im = np.transpose(color_im,(2,0,1))
        # Data Augmentation
        if self.augmentation:
            img, mask = augment_imgs_and_masks(x_img, color_im, rot_factor, scale_factor, trans_factor, flip)
        
        return x_img, color_im,y_img
#         if self.transforms:
#             augmented = self.transforms(image=x_img,mask=color_im)
#             img = augmented['image']
#             mask = augmented['mask']
#             return img, mask,y_img
# #         return x_img,color_im
```


```python
Adata_path_folder=sorted(os.listdir("./train/DICOM")) 
label_path_folder=sorted(os.listdir("./train/label"))

```


```python
#case 겹치지 않게 train,val 나누기
import glob
val_input_files=[]
val_label_files=[]
train_input_files=[]
train_label_files=[]
test_input_files=[]
test_label_files=[]

# for i in range(100):
#   if i>79:
#     val_input_files+=sorted(glob.glob("./train/DICOM/"+data_path_folder[i]+"/*.dcm",recursive=True))
#     val_label_files+=sorted(glob.glob("./train/Label/"+label_path_folder[i]+"/*.png",recursive=True))
#   else:
#     train_input_files+=sorted(glob.glob("./train/DICOM/"+data_path_folder[i]+"/*.dcm",recursive=True))
#     train_label_files+=sorted(glob.glob("./train/Label/"+label_path_folder[i]+"/*.png",recursive=True))   
```


```python
for i in range(102):
    if i==0:
        train_input_files+=sorted(glob.glob("./train/DICOM/"+Adata_path_folder[i]+"/*.npy",recursive=True))
        train_label_files+=sorted(glob.glob("./train/Label/"+label_path_folder[i]+"/*.npy",recursive=True))
    elif i<70:
        train_input_files+=sorted(glob.glob("./train/DICOM/"+Adata_path_folder[i]+"/*.dcm",recursive=True))
        train_label_files+=sorted(glob.glob("./train/Label/"+label_path_folder[i]+"/*.png",recursive=True))
    elif i<90 :
        val_input_files+=sorted(glob.glob("./train/DICOM/"+Adata_path_folder[i]+"/*.dcm",recursive=True))
        val_label_files+=sorted(glob.glob("./train/Label/"+label_path_folder[i]+"/*.png",recursive=True))
    elif i==101: 
        train_input_files+=sorted(glob.glob("./train/DICOM/"+Adata_path_folder[i]+"/*.npy",recursive=True))
        train_label_files+=sorted(glob.glob("./train/Label/"+label_path_folder[i]+"/*.npy",recursive=True))
    else:  
        test_input_files+=sorted(glob.glob("./train/DICOM/"+Adata_path_folder[i]+"/*.dcm",recursive=True))
        test_label_files+=sorted(glob.glob("./train/Label/"+label_path_folder[i]+"/*.png",recursive=True)) 
```


```python
len(train_input_files),len(val_input_files),len(test_input_files)
```




    (5749, 1280, 704)




```python
train_input_files = np.array(train_input_files)
train_label_files = np.array(train_label_files)

val_input_files = np.array(val_input_files)
val_label_files = np.array(val_label_files)

test_input_files = np.array(test_input_files)
test_label_files=np.array(test_label_files)
```


```python
train_dataset = MyDataset(train_input_files,train_label_files)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=8,shuffle=True)
val_dataset = MyDataset(val_input_files,val_label_files)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=8,shuffle=True)
```


```python
##input과 label이 맞나 확인
images,labels,a = next(iter(train_loader))
print(images.shape)
print(labels.shape)
print(labels[labels>=1])
plt.figure(figsize=(16,18))
plt.subplot(1,4,1)
plt.imshow(images[0][0],cmap='gray')
plt.subplot(1,4,2)
plt.imshow(labels[0][0])
plt.subplot(1,4,3)
plt.imshow(labels[0][1])
plt.subplot(1,4,4)
plt.imshow(a[0])
plt.show()
```

    torch.Size([8, 3, 512, 512])
    torch.Size([8, 2, 512, 512])
    tensor([1., 1., 1.,  ..., 1., 1., 1.], dtype=torch.float64)
    


![png](output_18_1.png)



```python
np.where(labels[0][0]>=0.1)
# len(labels[0][0]==0)
```




    (array([250, 250, 250, ..., 355, 355, 355], dtype=int64),
     array([192, 193, 194, ..., 367, 368, 369], dtype=int64))




```python
labels[0][0][306]
```




    tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.], dtype=torch.float64)




```python
def compute_per_channel_dice(input, target, epsilon=1e-5,ignore_index=None, weight=None):
    # assumes that input is a normalized probability
    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    # mask ignore_index if present
    if ignore_index is not None:
        mask = target.clone().ne_(ignore_index)
        mask.requires_grad = False

        input = input * mask
        target = target * mask

    input = flatten(input)
    target = flatten(target)

    # Compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    denominator = (input + target).sum(-1)
    return 2. * intersect / denominator.clamp(min=epsilon)

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order).contiguous()
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.view(C, -1)

class DiceLoss(nn.Module):
    """Computes Dice Loss, which just 1 - DiceCoefficient described above.
    Additionally allows per-class weights to be provided.
    """

    def __init__(self, epsilon=1e-5, weight=None, ignore_index=None, sigmoid_normalization=True,
                 skip_last_target=False):
        super(DiceLoss, self).__init__()
        if isinstance(weight, list):
            weight = torch.Tensor(weight)
            
        self.epsilon = epsilon
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index

        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)
        # if True skip the last channel in the target
        self.skip_last_target = skip_last_target

    def forward(self, input, target):
        # get probabilities from logits

        input = self.normalization(input)
        if self.weight is not None:
            weight = Variable(self.weight, requires_grad=False).to(input.device)
        else:
            weight = None

        if self.skip_last_target:
            target = target[:, :-1, ...]

        per_channel_dice = compute_per_channel_dice(input, target, epsilon=self.epsilon, ignore_index=self.ignore_index, weight=weight)
        # Average the Dice score across all channels/classes
        return torch.mean(1. - per_channel_dice)
```


```python
# pip install git+https://github.com/qubvel/segmentation_models.pytorch
```


```python
import segmentation_models_pytorch as smp
model = smp.FPN(  #DeepLabV3
    encoder_name="resnext101_32x8d",# choose encoder, e.g. mobilenet_v2 or efficientnet-b7 resnext101_32x8d,timm-res2net101_26w_4s     # use `imagenet` pre-trained weights for encoder initialization 
    encoder_weights="imagenet",
    in_channels=3,
    # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=2,                      # model output channels (number of classes in your dataset)
)
```


```python
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr


        self.enc1_1 = CBR2d(in_channels=3, out_channels=128)
        self.enc1_2 = CBR2d(in_channels=128, out_channels=128)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc2_2 = CBR2d(in_channels=256, out_channels=256)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc3_2 = CBR2d(in_channels=512, out_channels=512)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=512, out_channels=1024)
        self.enc4_2 = CBR2d(in_channels=1024, out_channels=1024)

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(in_channels=1024, out_channels=2048)
        

        self.dec5_1 = CBR2d(in_channels=2048, out_channels=1024)

        self.unpool4 = nn.ConvTranspose2d(in_channels=1024, out_channels=1024,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2 = CBR2d(in_channels=2 * 1024, out_channels=1024)
        self.dec4_1 = CBR2d(in_channels=1024, out_channels=512)

        self.unpool3 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = CBR2d(in_channels=2 * 512, out_channels=512)
        self.dec3_1 = CBR2d(in_channels=512, out_channels=256)

        self.unpool2 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR2d(in_channels=2 * 256, out_channels=256)
        self.dec2_1 = CBR2d(in_channels=256, out_channels=128)

        self.unpool1 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = CBR2d(in_channels=2 * 128, out_channels=128)
        self.dec1_1 = CBR2d(in_channels=128, out_channels=128)

        self.fc = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)
        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        x = self.fc(dec1_1)
        return x
```


```python
# model=UNet()
# # model
```


```python
# pip install monai
```


```python
# from monai.losses import DiceCELoss
# criterion = DiceCELoss(to_onehot_y=False, softmax=False)
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
# # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
```


```python
from sklearn.metrics import confusion_matrix  
 #mport numpy as np

def compute_iou(y_pred, y_true):
    y_pred=y_pred.detach().cpu()
    y_true=y_true.detach().cpu()
    # ytrue, ypred is a flatten vector
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    current = confusion_matrix(y_true, y_pred,labels=[0,1])
    # compute mean iou
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    return np.mean(IoU)
```


```python
sum([param.nelement() for param in model.parameters()])
```




    89350466




```python
# #model
# model.load_state_dict(torch.load('model_best_2.pt'))
```


```python
import torch.optim as optim
criterion =  DiceLoss(sigmoid_normalization=True)
# optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-8, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=6)

```


```python
n_epochs =100
cnt =0
valid_loss_min = np.Inf # track change in validation loss

# keep track of training and validation loss
train_loss = torch.zeros(n_epochs)
valid_loss = torch.zeros(n_epochs)
Iou=0
model.to(device)
for e in range(0, n_epochs):

   
    ###################
    # train the model #
    ###################
    model.train()
    for data, labels,a in tqdm(train_loader):
        # move tensors to GPU if CUDA is available
        data, labels = data.to(device), labels.to(device) #cpu에 있는 데이터를 gpu에 보냄
        # clear the gradients of all optimized variables
#         print(data.shape)
#         break
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        logits = model(data)
        # print(logits.shape)
        # print(labels.shape)
        # calculate the batch loss
        loss = criterion(logits, labels)
#         loss2 = criterion2(logits, labels)
#         loss=(loss+loss2)/2
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss[e] += loss.item()
        
        
#         z=logits.detach().cpu().numpy()
#         z = z.astype(np.uint8)
        cnt = cnt+1
        
        
        if cnt %50==0:
            
            logits = logits.sigmoid()
            logits = mask_binarization(logits.detach().cpu(), 0.5)
            iou = compute_iou(logits,labels)
            # iou=  get_iou(logits,labels)
            print(iou)
            # y=torch.squeeze(labels[0])
            y=logits[0].detach().cpu().numpy()
            # x=data[0].detach().cpu().numpy()
            x=labels[0].detach().cpu().numpy()
            #y=labels[0].numpy()
            plt.figure(figsize=(16,18))
            plt.subplot(1,5,1)
            plt.imshow(x[0])
            plt.subplot(1,5,2)
            plt.imshow(x[1])
            plt.subplot(1,5,3)
            plt.imshow(y[0])
            plt.subplot(1,5,4)
            plt.imshow(y[1])
            plt.subplot(1,5,5)
            plt.imshow(a[0])
            plt.show()

    
    train_loss[e] /= len(train_loader)
    #torch.save(model.state_dict(), 'model_.pt')
        
    ######################    
    # validate the model #
    ######################
    with torch.no_grad(): 
        model.eval()
        for data, labels,a in tqdm(val_loader):
            # move tensors to GPU if CUDA is available
            data, labels = data.to(device), labels.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            logits = model(data)
            # calculate the batch loss
            loss = criterion(logits, labels)
#             loss2 = criterion2(logits, labels)
#             loss=(loss+loss2)/2
            # update average validation loss 
            valid_loss[e] += loss.item()

    
    # calculate average losses
    valid_loss[e] /= len(val_loader)
    # scheduler.step(valid_loss[e])    
    # print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        e, train_loss[e], valid_loss[e]))
    
    # save model if validation loss has decreased
    if valid_loss[e] <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss[e]))
        torch.save(model.state_dict(), 'model_best_2.pt')
        valid_loss_min = valid_loss[e]
```


      0%|          | 0/719 [00:00<?, ?it/s]


    0.730210056638347
    


![png](output_32_2.png)


    0.5606632277427123
    


![png](output_32_4.png)


    0.7076403817148942
    


![png](output_32_6.png)


    0.7782949327460411
    


![png](output_32_8.png)


    0.6014649129031372
    


![png](output_32_10.png)


    0.5711671079256747
    


![png](output_32_12.png)


    0.617816451273619
    


![png](output_32_14.png)


    0.5830891279378511
    


![png](output_32_16.png)


    0.803961154441334
    


![png](output_32_18.png)


    0.7415374631348512
    


![png](output_32_20.png)


    0.6960588900922763
    


![png](output_32_22.png)


    0.6384012853185572
    


![png](output_32_24.png)


    0.8869818247396246
    


![png](output_32_26.png)


    0.7296441177770115
    


![png](output_32_28.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 0 	Training Loss: 0.561646 	Validation Loss: 0.587100
    Validation loss decreased (inf --> 0.587100).  Saving model ...
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.8011646349635465
    


![png](output_32_33.png)


    0.8832904172910321
    


![png](output_32_35.png)


    0.7498658234686104
    


![png](output_32_37.png)


    0.9056659445674136
    


![png](output_32_39.png)


    0.7020550737142491
    


![png](output_32_41.png)


    0.8000452165667041
    


![png](output_32_43.png)


    0.7860323000531216
    


![png](output_32_45.png)


    0.8125882657728157
    


![png](output_32_47.png)


    0.8602335466266555
    


![png](output_32_49.png)


    0.8884385153722794
    


![png](output_32_51.png)


    0.8824523574423186
    


![png](output_32_53.png)


    0.7912412329846208
    


![png](output_32_55.png)


    0.8322638606080529
    


![png](output_32_57.png)


    0.7744584940539644
    


![png](output_32_59.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 1 	Training Loss: 0.435473 	Validation Loss: 0.496759
    Validation loss decreased (0.587100 --> 0.496759).  Saving model ...
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.8643511988423366
    


![png](output_32_64.png)


    0.9128655320853657
    


![png](output_32_66.png)


    0.7777311438767338
    


![png](output_32_68.png)


    0.7857187626552902
    


![png](output_32_70.png)


    0.8745327893240159
    


![png](output_32_72.png)


    0.838724979619063
    


![png](output_32_74.png)


    0.8154140967778233
    


![png](output_32_76.png)


    0.7934606405632081
    


![png](output_32_78.png)


    0.8503257144289638
    


![png](output_32_80.png)


    0.9082075537735084
    


![png](output_32_82.png)


    0.8810808171514217
    


![png](output_32_84.png)


    0.8979345145134234
    


![png](output_32_86.png)


    0.8506315518475128
    


![png](output_32_88.png)


    0.8897468928373837
    


![png](output_32_90.png)


    0.799349302689393
    


![png](output_32_92.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 2 	Training Loss: 0.398958 	Validation Loss: 0.500103
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.8912366200794266
    


![png](output_32_97.png)


    0.65579355080429
    


![png](output_32_99.png)


    0.8979003817753867
    


![png](output_32_101.png)


    0.8216064367628428
    


![png](output_32_103.png)


    0.8357493120031493
    


![png](output_32_105.png)


    0.8665345365575324
    


![png](output_32_107.png)


    0.9092529264705247
    


![png](output_32_109.png)


    0.8781631185546162
    


![png](output_32_111.png)


    0.8578711425437608
    


![png](output_32_113.png)


    0.9022000875871794
    


![png](output_32_115.png)


    0.7800709566666102
    


![png](output_32_117.png)


    0.9150507340489991
    


![png](output_32_119.png)


    0.8904722589132175
    


![png](output_32_121.png)


    0.8632046043354615
    


![png](output_32_123.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 3 	Training Loss: 0.375796 	Validation Loss: 0.462885
    Validation loss decreased (0.496759 --> 0.462885).  Saving model ...
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9191073606044387
    


![png](output_32_128.png)


    0.8531212860300278
    


![png](output_32_130.png)


    0.8824699113862837
    


![png](output_32_132.png)


    0.8780833363254863
    


![png](output_32_134.png)


    0.8890502625185169
    


![png](output_32_136.png)


    0.8681073145729417
    


![png](output_32_138.png)


    0.76216260517947
    


![png](output_32_140.png)


    0.941449930466381
    


![png](output_32_142.png)


    0.8785129230199606
    


![png](output_32_144.png)


    0.8077736374623383
    


![png](output_32_146.png)


    0.836480053284063
    


![png](output_32_148.png)


    0.9513243994010447
    


![png](output_32_150.png)


    0.8507384105930909
    


![png](output_32_152.png)


    0.8942245959477575
    


![png](output_32_154.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 4 	Training Loss: 0.357268 	Validation Loss: 0.469093
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.8565057428455577
    


![png](output_32_159.png)


    0.901337573725024
    


![png](output_32_161.png)


    0.9354254773498885
    


![png](output_32_163.png)


    0.9000894073071187
    


![png](output_32_165.png)


    0.8816549268499134
    


![png](output_32_167.png)


    0.8925400493482258
    


![png](output_32_169.png)


    0.9314575162415217
    


![png](output_32_171.png)


    0.8812901922271352
    


![png](output_32_173.png)


    0.8453050670165405
    


![png](output_32_175.png)


    0.8972219007443502
    


![png](output_32_177.png)


    0.8934574392781456
    


![png](output_32_179.png)


    0.8550674923606643
    


![png](output_32_181.png)


    0.9202071258611731
    


![png](output_32_183.png)


    0.8850094356972669
    


![png](output_32_185.png)


    0.9249820703798067
    


![png](output_32_187.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 5 	Training Loss: 0.327958 	Validation Loss: 0.502284
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9361156496119883
    


![png](output_32_192.png)


    0.889603428651971
    


![png](output_32_194.png)


    0.9479593515210102
    


![png](output_32_196.png)


    0.9346451543591903
    


![png](output_32_198.png)


    0.8950493309314325
    


![png](output_32_200.png)


    0.9287822484564294
    


![png](output_32_202.png)


    0.91590077286462
    


![png](output_32_204.png)


    0.9055683654736687
    


![png](output_32_206.png)


    0.8853418278061337
    


![png](output_32_208.png)


    0.8879875151423607
    


![png](output_32_210.png)


    0.8197198124733566
    


![png](output_32_212.png)


    0.917157784382967
    


![png](output_32_214.png)


    0.8808013017305024
    


![png](output_32_216.png)


    0.9332825128786119
    


![png](output_32_218.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 6 	Training Loss: 0.321759 	Validation Loss: 0.413945
    Validation loss decreased (0.462885 --> 0.413945).  Saving model ...
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9215628483758099
    


![png](output_32_223.png)


    0.8770089782984174
    


![png](output_32_225.png)


    0.9001453066212319
    


![png](output_32_227.png)


    0.917724028603508
    


![png](output_32_229.png)


    0.9135253093444143
    


![png](output_32_231.png)


    0.90330608687867
    


![png](output_32_233.png)


    0.9169200404541213
    


![png](output_32_235.png)


    0.906361357677756
    


![png](output_32_237.png)


    0.8336436604167782
    


![png](output_32_239.png)


    0.9107406995515988
    


![png](output_32_241.png)


    0.9590347728639537
    


![png](output_32_243.png)


    0.9166778875046842
    


![png](output_32_245.png)


    0.8852785489710622
    


![png](output_32_247.png)


    0.7965164999892771
    


![png](output_32_249.png)


    0.912474869464724
    


![png](output_32_251.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 7 	Training Loss: 0.301964 	Validation Loss: 0.372858
    Validation loss decreased (0.413945 --> 0.372858).  Saving model ...
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.8950107386999357
    


![png](output_32_256.png)


    0.9309879029494834
    


![png](output_32_258.png)


    0.8844347719718872
    


![png](output_32_260.png)


    0.8668938913946768
    


![png](output_32_262.png)


    0.890506017880976
    


![png](output_32_264.png)


    0.9104921244229984
    


![png](output_32_266.png)


    0.8795385524340036
    


![png](output_32_268.png)


    0.9353254018573184
    


![png](output_32_270.png)


    0.922297741980306
    


![png](output_32_272.png)


    0.7676897165486217
    


![png](output_32_274.png)


    0.8591147364857536
    


![png](output_32_276.png)


    0.8394354541611939
    


![png](output_32_278.png)


    0.9008784753440788
    


![png](output_32_280.png)


    0.9272644598682362
    


![png](output_32_282.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 8 	Training Loss: 0.284363 	Validation Loss: 0.429222
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.8995836897146927
    


![png](output_32_287.png)


    0.8918329981409165
    


![png](output_32_289.png)


    0.9524253714540258
    


![png](output_32_291.png)


    0.7494704628947663
    


![png](output_32_293.png)


    0.9270190039885442
    


![png](output_32_295.png)


    0.947775102665781
    


![png](output_32_297.png)


    0.9476284329408051
    


![png](output_32_299.png)


    0.8912645331789546
    


![png](output_32_301.png)


    0.8882659851556818
    


![png](output_32_303.png)


    0.8922773993000862
    


![png](output_32_305.png)


    0.8546819067297869
    


![png](output_32_307.png)


    0.8462259974357682
    


![png](output_32_309.png)


    0.9099590229135543
    


![png](output_32_311.png)


    0.8475390121440713
    


![png](output_32_313.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 9 	Training Loss: 0.282134 	Validation Loss: 0.460613
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9113250746351855
    


![png](output_32_318.png)


    0.9298402196813336
    


![png](output_32_320.png)


    0.9317514794752908
    


![png](output_32_322.png)


    0.8577962944572697
    


![png](output_32_324.png)


    0.73460020646094
    


![png](output_32_326.png)


    0.8807543336942506
    


![png](output_32_328.png)


    0.9424431930593195
    


![png](output_32_330.png)


    0.9096602370272009
    


![png](output_32_332.png)


    0.9246576901126456
    


![png](output_32_334.png)


    0.9287167515868159
    


![png](output_32_336.png)


    0.9298799329087162
    


![png](output_32_338.png)


    0.8843540460330361
    


![png](output_32_340.png)


    0.9462236663311132
    


![png](output_32_342.png)


    0.9424773719674903
    


![png](output_32_344.png)


    0.9506336291664605
    


![png](output_32_346.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 10 	Training Loss: 0.262792 	Validation Loss: 0.393389
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9180626831850263
    


![png](output_32_351.png)


    0.94758434058151
    


![png](output_32_353.png)


    0.9032942336193089
    


![png](output_32_355.png)


    0.9119088726608497
    


![png](output_32_357.png)


    0.8436217352719906
    


![png](output_32_359.png)


    0.9173367059226056
    


![png](output_32_361.png)


    0.9283629810220946
    


![png](output_32_363.png)


    0.932087784753807
    


![png](output_32_365.png)


    0.9012435479457069
    


![png](output_32_367.png)


    0.9186189031868948
    


![png](output_32_369.png)


    0.902427653357661
    


![png](output_32_371.png)


    0.8996005369183293
    


![png](output_32_373.png)


    0.914861365686942
    


![png](output_32_375.png)


    0.9564663866695364
    


![png](output_32_377.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 11 	Training Loss: 0.276207 	Validation Loss: 0.402931
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.7685152299444737
    


![png](output_32_382.png)


    0.8938148619943116
    


![png](output_32_384.png)


    0.9332369955088271
    


![png](output_32_386.png)


    0.8803141817633329
    


![png](output_32_388.png)


    0.8340409363967589
    


![png](output_32_390.png)


    0.9232538345784618
    


![png](output_32_392.png)


    0.8848107813332019
    


![png](output_32_394.png)


    0.9059579718945565
    


![png](output_32_396.png)


    0.9068659814591813
    


![png](output_32_398.png)


    0.9503638625280719
    


![png](output_32_400.png)


    0.9092560574099839
    


![png](output_32_402.png)


    0.9146727578076441
    


![png](output_32_404.png)


    0.9027451838655853
    


![png](output_32_406.png)


    0.9043681897612799
    


![png](output_32_408.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 12 	Training Loss: 0.259462 	Validation Loss: 0.381792
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.7046571418955228
    


![png](output_32_413.png)


    0.9532820609111379
    


![png](output_32_415.png)


    0.9020338776430656
    


![png](output_32_417.png)


    0.8942961656749544
    


![png](output_32_419.png)


    0.9124960988455275
    


![png](output_32_421.png)


    0.9560110106620276
    


![png](output_32_423.png)


    0.8845243641832572
    


![png](output_32_425.png)


    0.909639450831709
    


![png](output_32_427.png)


    0.9485299498432019
    


![png](output_32_429.png)


    0.9454350707557279
    


![png](output_32_431.png)


    0.9437348496150988
    


![png](output_32_433.png)


    0.9178303471128553
    


![png](output_32_435.png)


    0.9184748598231451
    


![png](output_32_437.png)


    0.9251000932999953
    


![png](output_32_439.png)


    0.8212046483513022
    


![png](output_32_441.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 13 	Training Loss: 0.240865 	Validation Loss: 0.364508
    Validation loss decreased (0.372858 --> 0.364508).  Saving model ...
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9253777898982638
    


![png](output_32_446.png)


    0.9078045108199709
    


![png](output_32_448.png)


    0.962293296570037
    


![png](output_32_450.png)


    0.9086397291578734
    


![png](output_32_452.png)


    0.9238131122779455
    


![png](output_32_454.png)


    0.9478185274704831
    


![png](output_32_456.png)


    0.9064030271122498
    


![png](output_32_458.png)


    0.9337512244934352
    


![png](output_32_460.png)


    0.9212936808995174
    


![png](output_32_462.png)


    0.9115621953159021
    


![png](output_32_464.png)


    0.9348636027477233
    


![png](output_32_466.png)


    0.8614633481853136
    


![png](output_32_468.png)


    0.9559660870763108
    


![png](output_32_470.png)


    0.9489293713677758
    


![png](output_32_472.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 14 	Training Loss: 0.247326 	Validation Loss: 0.377686
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9209536325571066
    


![png](output_32_477.png)


    0.954732413073919
    


![png](output_32_479.png)


    0.8799231621240127
    


![png](output_32_481.png)


    0.9155544683145403
    


![png](output_32_483.png)


    0.9208102794273145
    


![png](output_32_485.png)


    0.9197397184979792
    


![png](output_32_487.png)


    0.9024601998708321
    


![png](output_32_489.png)


    0.9449988725360011
    


![png](output_32_491.png)


    0.9137646210264059
    


![png](output_32_493.png)


    0.9652303725181945
    


![png](output_32_495.png)


    0.8403539995547418
    


![png](output_32_497.png)


    0.9413358215061663
    


![png](output_32_499.png)


    0.7484889899881626
    


![png](output_32_501.png)


    0.9601488484150784
    


![png](output_32_503.png)


    0.9184203669810416
    


![png](output_32_505.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 15 	Training Loss: 0.250594 	Validation Loss: 0.372829
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9592237317171206
    


![png](output_32_510.png)


    0.9364045121799214
    


![png](output_32_512.png)


    0.9343979009490078
    


![png](output_32_514.png)


    0.8583286446360732
    


![png](output_32_516.png)


    0.9441452865933211
    


![png](output_32_518.png)


    0.9311422242463914
    


![png](output_32_520.png)


    0.9325273491691618
    


![png](output_32_522.png)


    0.9431374583221093
    


![png](output_32_524.png)


    0.8928941152796357
    


![png](output_32_526.png)


    0.9657835151545582
    


![png](output_32_528.png)


    0.9628922632678167
    


![png](output_32_530.png)


    0.9273810408147286
    


![png](output_32_532.png)


    0.9300972781698588
    


![png](output_32_534.png)


    0.9171073198290275
    


![png](output_32_536.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 16 	Training Loss: 0.218324 	Validation Loss: 0.497346
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.7481565460513008
    


![png](output_32_541.png)


    0.9346854811821222
    


![png](output_32_543.png)


    0.9229679197409495
    


![png](output_32_545.png)


    0.9500194990150801
    


![png](output_32_547.png)


    0.9533147406567316
    


![png](output_32_549.png)


    0.9438468021299704
    


![png](output_32_551.png)


    0.8639949097611118
    


![png](output_32_553.png)


    0.9524136678885735
    


![png](output_32_555.png)


    0.9418002953718345
    


![png](output_32_557.png)


    0.8517003599467496
    


![png](output_32_559.png)


    0.9116358049124056
    


![png](output_32_561.png)


    0.9661048173768783
    


![png](output_32_563.png)


    0.9628796774514872
    


![png](output_32_565.png)


    0.9348948750436723
    


![png](output_32_567.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 17 	Training Loss: 0.225619 	Validation Loss: 0.439952
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9489604127016198
    


![png](output_32_572.png)


    0.9628185387107804
    


![png](output_32_574.png)


    0.943990590611985
    


![png](output_32_576.png)


    0.9571389613971707
    


![png](output_32_578.png)


    0.87375159950609
    


![png](output_32_580.png)


    0.9721844748248334
    


![png](output_32_582.png)


    0.862025102445902
    


![png](output_32_584.png)


    0.9299560584082045
    


![png](output_32_586.png)


    0.922792804000899
    


![png](output_32_588.png)


    0.9542427913129576
    


![png](output_32_590.png)


    0.9582590201397925
    


![png](output_32_592.png)


    0.9205258385559543
    


![png](output_32_594.png)


    0.9065905782093275
    


![png](output_32_596.png)


    0.9611306026662553
    


![png](output_32_598.png)


    0.8065828585866655
    


![png](output_32_600.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 18 	Training Loss: 0.208850 	Validation Loss: 0.474559
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.971189287571083
    


![png](output_32_605.png)


    0.9316181117840281
    


![png](output_32_607.png)


    0.8201068713081263
    


![png](output_32_609.png)


    0.9123265934077465
    


![png](output_32_611.png)


    0.9546142177831329
    


![png](output_32_613.png)


    0.946211915227287
    


![png](output_32_615.png)


    0.9524343604648876
    


![png](output_32_617.png)


    0.9395519722371009
    


![png](output_32_619.png)


    0.9577156238579386
    


![png](output_32_621.png)


    0.9506297473994874
    


![png](output_32_623.png)


    0.9352385031855336
    


![png](output_32_625.png)


    0.959550804193811
    


![png](output_32_627.png)


    0.9562671546496613
    


![png](output_32_629.png)


    0.9504928356642481
    


![png](output_32_631.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 19 	Training Loss: 0.221825 	Validation Loss: 0.368035
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9092112522532751
    


![png](output_32_636.png)


    0.93932108994079
    


![png](output_32_638.png)


    0.9685677966406321
    


![png](output_32_640.png)


    0.9669472897587679
    


![png](output_32_642.png)


    0.9527930351836169
    


![png](output_32_644.png)


    0.9459740014178042
    


![png](output_32_646.png)


    0.9316131699608682
    


![png](output_32_648.png)


    0.9326940636019174
    


![png](output_32_650.png)


    0.9487427789098068
    


![png](output_32_652.png)


    0.957095885885187
    


![png](output_32_654.png)


    0.9290242192225382
    


![png](output_32_656.png)


    0.8995453349455479
    


![png](output_32_658.png)


    0.9652365031875549
    


![png](output_32_660.png)


    0.882182844998629
    


![png](output_32_662.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 20 	Training Loss: 0.199936 	Validation Loss: 0.365119
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.8953866995436248
    


![png](output_32_667.png)


    0.9575377036992689
    


![png](output_32_669.png)


    0.9433902683355906
    


![png](output_32_671.png)


    0.945256613797478
    


![png](output_32_673.png)


    0.9685472725553215
    


![png](output_32_675.png)


    0.9387833481915766
    


![png](output_32_677.png)


    0.9532508506285566
    


![png](output_32_679.png)


    0.9316039969841832
    


![png](output_32_681.png)


    0.9577662027897049
    


![png](output_32_683.png)


    0.907798281004514
    


![png](output_32_685.png)


    0.8943703961543628
    


![png](output_32_687.png)


    0.932990833634466
    


![png](output_32_689.png)


    0.9349456261524733
    


![png](output_32_691.png)


    0.9561505876870444
    


![png](output_32_693.png)


    0.9222610429417615
    


![png](output_32_695.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 21 	Training Loss: 0.201331 	Validation Loss: 0.361974
    Validation loss decreased (0.364508 --> 0.361974).  Saving model ...
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9433384064679806
    


![png](output_32_700.png)


    0.9464001430799294
    


![png](output_32_702.png)


    0.9621663395906953
    


![png](output_32_704.png)


    0.9419953892131256
    


![png](output_32_706.png)


    0.9192308102871063
    


![png](output_32_708.png)


    0.9196394845859945
    


![png](output_32_710.png)


    0.9512388229197685
    


![png](output_32_712.png)


    0.9295570075132494
    


![png](output_32_714.png)


    0.9742818589839655
    


![png](output_32_716.png)


    0.9514811742139558
    


![png](output_32_718.png)


    0.9451242037126572
    


![png](output_32_720.png)


    0.961460661797328
    


![png](output_32_722.png)


    0.8981263872416831
    


![png](output_32_724.png)


    0.9556622437787405
    


![png](output_32_726.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 22 	Training Loss: 0.190989 	Validation Loss: 0.390246
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9810297163011563
    


![png](output_32_731.png)


    0.915489069008792
    


![png](output_32_733.png)


    0.9277236173365578
    


![png](output_32_735.png)


    0.9585352181806266
    


![png](output_32_737.png)


    0.9457560506564199
    


![png](output_32_739.png)


    0.9427793442624345
    


![png](output_32_741.png)


    0.921637298892593
    


![png](output_32_743.png)


    0.8987479934716829
    


![png](output_32_745.png)


    0.9216513946030407
    


![png](output_32_747.png)


    0.9459317377902992
    


![png](output_32_749.png)


    0.9609076131522163
    


![png](output_32_751.png)


    0.94939486079971
    


![png](output_32_753.png)


    0.9461370406959165
    


![png](output_32_755.png)


    0.9458462421734151
    


![png](output_32_757.png)


    0.9389969242767722
    


![png](output_32_759.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 23 	Training Loss: 0.195012 	Validation Loss: 0.373367
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9401590497682508
    


![png](output_32_764.png)


    0.9401701171067325
    


![png](output_32_766.png)


    0.9262692640131849
    


![png](output_32_768.png)


    0.9552846588839472
    


![png](output_32_770.png)


    0.9563039775214863
    


![png](output_32_772.png)


    0.9629314497432293
    


![png](output_32_774.png)


    0.9411695589554823
    


![png](output_32_776.png)


    0.929834080622232
    


![png](output_32_778.png)


    0.85337389160696
    


![png](output_32_780.png)


    0.8672081818618433
    


![png](output_32_782.png)


    0.9467493448431614
    


![png](output_32_784.png)


    0.9184848040351883
    


![png](output_32_786.png)


    0.9450596664299449
    


![png](output_32_788.png)


    0.9647114337824781
    


![png](output_32_790.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 24 	Training Loss: 0.187108 	Validation Loss: 0.340931
    Validation loss decreased (0.361974 --> 0.340931).  Saving model ...
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9596846372343788
    


![png](output_32_795.png)


    0.9523722176528533
    


![png](output_32_797.png)


    0.951774383153293
    


![png](output_32_799.png)


    0.9543005575973973
    


![png](output_32_801.png)


    0.8905261556418664
    


![png](output_32_803.png)


    0.9606928084701638
    


![png](output_32_805.png)


    0.8202525405154573
    


![png](output_32_807.png)


    0.8846461554768434
    


![png](output_32_809.png)


    0.9722898181561732
    


![png](output_32_811.png)


    0.9348131851002519
    


![png](output_32_813.png)


    0.957603532802785
    


![png](output_32_815.png)


    0.904269559122443
    


![png](output_32_817.png)


    0.9699865459309797
    


![png](output_32_819.png)


    0.9377481700581418
    


![png](output_32_821.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 25 	Training Loss: 0.190333 	Validation Loss: 0.323970
    Validation loss decreased (0.340931 --> 0.323970).  Saving model ...
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9514689492321786
    


![png](output_32_826.png)


    0.9448992845165884
    


![png](output_32_828.png)


    0.9649394422323962
    


![png](output_32_830.png)


    0.9413939000535241
    


![png](output_32_832.png)


    0.9056346518880106
    


![png](output_32_834.png)


    0.9634660941728608
    


![png](output_32_836.png)


    0.9535792353841142
    


![png](output_32_838.png)


    0.9634567025501972
    


![png](output_32_840.png)


    0.9468203984971629
    


![png](output_32_842.png)


    0.9325074336296324
    


![png](output_32_844.png)


    0.9658073184442397
    


![png](output_32_846.png)


    0.9435529079511341
    


![png](output_32_848.png)


    0.9487821481583806
    


![png](output_32_850.png)


    0.9132778024179753
    


![png](output_32_852.png)


    0.9621122323207957
    


![png](output_32_854.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 26 	Training Loss: 0.185931 	Validation Loss: 0.367894
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.8709387225679601
    


![png](output_32_859.png)


    0.9384450425761609
    


![png](output_32_861.png)


    0.9651042192095978
    


![png](output_32_863.png)


    0.9448602942954791
    


![png](output_32_865.png)


    0.8412470705404207
    


![png](output_32_867.png)


    0.9764389679637226
    


![png](output_32_869.png)


    0.9154994457883872
    


![png](output_32_871.png)


    0.9769828231829163
    


![png](output_32_873.png)


    0.9156085542435977
    


![png](output_32_875.png)


    0.891741988628932
    


![png](output_32_877.png)


    0.9550262306934945
    


![png](output_32_879.png)


    0.9350939225617019
    


![png](output_32_881.png)


    0.96125194417243
    


![png](output_32_883.png)


    0.9215576440938671
    


![png](output_32_885.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 27 	Training Loss: 0.182242 	Validation Loss: 0.452701
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9456650308267032
    


![png](output_32_890.png)


    0.9386446916887938
    


![png](output_32_892.png)


    0.9260126667790772
    


![png](output_32_894.png)


    0.9501205625897872
    


![png](output_32_896.png)


    0.9284730385757273
    


![png](output_32_898.png)


    0.9345636941655825
    


![png](output_32_900.png)


    0.9596785122146715
    


![png](output_32_902.png)


    0.9502770534521792
    


![png](output_32_904.png)


    0.959315676090685
    


![png](output_32_906.png)


    0.9292655674177144
    


![png](output_32_908.png)


    0.96158482888216
    


![png](output_32_910.png)


    0.8959845568611244
    


![png](output_32_912.png)


    0.9561374284547843
    


![png](output_32_914.png)


    0.9106558440953152
    


![png](output_32_916.png)


    0.9500572892218753
    


![png](output_32_918.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 28 	Training Loss: 0.178795 	Validation Loss: 0.387601
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9629854834291177
    


![png](output_32_923.png)


    0.9545612568930523
    


![png](output_32_925.png)


    0.9132114407127083
    


![png](output_32_927.png)


    0.9645376583145421
    


![png](output_32_929.png)


    0.9628388602862492
    


![png](output_32_931.png)


    0.923652037099481
    


![png](output_32_933.png)


    0.962857975681115
    


![png](output_32_935.png)


    0.9209426607891945
    


![png](output_32_937.png)


    0.9001381417880683
    


![png](output_32_939.png)


    0.9411588723325003
    


![png](output_32_941.png)


    0.9287847654464918
    


![png](output_32_943.png)


    0.9460242279211215
    


![png](output_32_945.png)


    0.9309008963491749
    


![png](output_32_947.png)


    0.9514333677345532
    


![png](output_32_949.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 29 	Training Loss: 0.179109 	Validation Loss: 0.378226
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.943040466106231
    


![png](output_32_954.png)


    0.9570682588197222
    


![png](output_32_956.png)


    0.9313489340974108
    


![png](output_32_958.png)


    0.7717865013224208
    


![png](output_32_960.png)


    0.9615951771777984
    


![png](output_32_962.png)


    0.9481872114813606
    


![png](output_32_964.png)


    0.9627271321325579
    


![png](output_32_966.png)


    0.9629836591479743
    


![png](output_32_968.png)


    0.954255081985657
    


![png](output_32_970.png)


    0.9763263974946162
    


![png](output_32_972.png)


    0.9386854774151672
    


![png](output_32_974.png)


    0.9420145597225942
    


![png](output_32_976.png)


    0.9618258084422607
    


![png](output_32_978.png)


    0.9580913887390088
    


![png](output_32_980.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 30 	Training Loss: 0.169888 	Validation Loss: 0.329633
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9606878053702742
    


![png](output_32_985.png)


    0.9403747986515593
    


![png](output_32_987.png)


    0.9636464895858601
    


![png](output_32_989.png)


    0.8965200914072757
    


![png](output_32_991.png)


    0.9565766301349649
    


![png](output_32_993.png)


    0.9504208950452444
    


![png](output_32_995.png)


    0.9766133449095618
    


![png](output_32_997.png)


    0.9568296235242457
    


![png](output_32_999.png)


    0.9254065241090609
    


![png](output_32_1001.png)


    0.9618520053458794
    


![png](output_32_1003.png)


    0.8741078170531028
    


![png](output_32_1005.png)


    0.889403708426515
    


![png](output_32_1007.png)


    0.9540309240908935
    


![png](output_32_1009.png)


    0.9419129459109667
    


![png](output_32_1011.png)


    0.9604604594359276
    


![png](output_32_1013.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 31 	Training Loss: 0.174909 	Validation Loss: 0.379130
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.8958385034585314
    


![png](output_32_1018.png)


    0.9666230351314071
    


![png](output_32_1020.png)


    0.9344105715743994
    


![png](output_32_1022.png)


    0.9603546405444552
    


![png](output_32_1024.png)


    0.9586114972423009
    


![png](output_32_1026.png)


    0.9682009362578706
    


![png](output_32_1028.png)


    0.9747983326763084
    


![png](output_32_1030.png)


    0.9612230008426703
    


![png](output_32_1032.png)


    0.9733967610702854
    


![png](output_32_1034.png)


    0.9492311879131132
    


![png](output_32_1036.png)


    0.9419103430635563
    


![png](output_32_1038.png)


    0.9762090243887571
    


![png](output_32_1040.png)


    0.967568119922277
    


![png](output_32_1042.png)


    0.964500551765207
    


![png](output_32_1044.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 32 	Training Loss: 0.167077 	Validation Loss: 0.382773
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.8941530057706665
    


![png](output_32_1049.png)


    0.9427546507549232
    


![png](output_32_1051.png)


    0.9554686697633425
    


![png](output_32_1053.png)


    0.9734813865405907
    


![png](output_32_1055.png)


    0.9514397265012606
    


![png](output_32_1057.png)


    0.9622343912337122
    


![png](output_32_1059.png)


    0.8960154468309436
    


![png](output_32_1061.png)


    0.923558455056311
    


![png](output_32_1063.png)


    0.9688232444992009
    


![png](output_32_1065.png)


    0.9659625343383403
    


![png](output_32_1067.png)


    0.9676279827818908
    


![png](output_32_1069.png)


    0.9632818661647636
    


![png](output_32_1071.png)


    0.961015762666293
    


![png](output_32_1073.png)


    0.8911262095705532
    


![png](output_32_1075.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 33 	Training Loss: 0.161015 	Validation Loss: 0.366897
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.8800243793766955
    


![png](output_32_1080.png)


    0.9440485774655725
    


![png](output_32_1082.png)


    0.9582649051197192
    


![png](output_32_1084.png)


    0.8786045631127564
    


![png](output_32_1086.png)


    0.9660606612319806
    


![png](output_32_1088.png)


    0.9713650394508615
    


![png](output_32_1090.png)


    0.9415895316622493
    


![png](output_32_1092.png)


    0.9179419040032096
    


![png](output_32_1094.png)


    0.964814290221953
    


![png](output_32_1096.png)


    0.9646778751931113
    


![png](output_32_1098.png)


    0.9404246133887095
    


![png](output_32_1100.png)


    0.9177165131878251
    


![png](output_32_1102.png)


    0.9510604333951556
    


![png](output_32_1104.png)


    0.9667741527440052
    


![png](output_32_1106.png)


    0.9613784664898654
    


![png](output_32_1108.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 34 	Training Loss: 0.166585 	Validation Loss: 0.372763
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9492924808945855
    


![png](output_32_1113.png)


    0.9614484946577366
    


![png](output_32_1115.png)


    0.9698776351050462
    


![png](output_32_1117.png)


    0.9231772998948549
    


![png](output_32_1119.png)


    0.9635295113233844
    


![png](output_32_1121.png)


    0.8898415978551826
    


![png](output_32_1123.png)


    0.952902855203698
    


![png](output_32_1125.png)


    0.9637665699588078
    


![png](output_32_1127.png)


    0.947435378774591
    


![png](output_32_1129.png)


    0.9636583941266138
    


![png](output_32_1131.png)


    0.9383156708117395
    


![png](output_32_1133.png)


    0.9563077104528946
    


![png](output_32_1135.png)


    0.9571853928676098
    


![png](output_32_1137.png)


    0.9300592197343114
    


![png](output_32_1139.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 35 	Training Loss: 0.169682 	Validation Loss: 0.395612
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.975630969300209
    


![png](output_32_1144.png)


    0.9646674913072679
    


![png](output_32_1146.png)


    0.9494527218855371
    


![png](output_32_1148.png)


    0.9687282411594971
    


![png](output_32_1150.png)


    0.9272059793439147
    


![png](output_32_1152.png)


    0.9638494475737478
    


![png](output_32_1154.png)


    0.8418897957320499
    


![png](output_32_1156.png)


    0.9337408120921877
    


![png](output_32_1158.png)


    0.9618357586915672
    


![png](output_32_1160.png)


    0.8993058691957458
    


![png](output_32_1162.png)


    0.9513061114708425
    


![png](output_32_1164.png)


    0.9379771284096815
    


![png](output_32_1166.png)


    0.9784168610499577
    


![png](output_32_1168.png)


    0.9598434534658316
    


![png](output_32_1170.png)


    0.9627963123163941
    


![png](output_32_1172.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 36 	Training Loss: 0.155234 	Validation Loss: 0.367818
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9462063864023567
    


![png](output_32_1177.png)


    0.9479553382402783
    


![png](output_32_1179.png)


    0.9707595711930961
    


![png](output_32_1181.png)


    0.9781975076523535
    


![png](output_32_1183.png)


    0.9411941307624121
    


![png](output_32_1185.png)


    0.9614432413762594
    


![png](output_32_1187.png)


    0.9603568348762666
    


![png](output_32_1189.png)


    0.9537110855027089
    


![png](output_32_1191.png)


    0.9304102674811883
    


![png](output_32_1193.png)


    0.92647575106652
    


![png](output_32_1195.png)


    0.9559970018224775
    


![png](output_32_1197.png)


    0.9639164041659433
    


![png](output_32_1199.png)


    0.9505300763007469
    


![png](output_32_1201.png)


    0.9043240769230125
    


![png](output_32_1203.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 37 	Training Loss: 0.160118 	Validation Loss: 0.400645
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.936373989024955
    


![png](output_32_1208.png)


    0.9489341093038843
    


![png](output_32_1210.png)


    0.9672716979175836
    


![png](output_32_1212.png)


    0.9344557807452958
    


![png](output_32_1214.png)


    0.963447251554667
    


![png](output_32_1216.png)


    0.9455963289733682
    


![png](output_32_1218.png)


    0.920332020863277
    


![png](output_32_1220.png)


    0.9483404755278704
    


![png](output_32_1222.png)


    0.8996139057286456
    


![png](output_32_1224.png)


    0.9697648983334073
    


![png](output_32_1226.png)


    0.9398658130358676
    


![png](output_32_1228.png)


    0.9552327779388365
    


![png](output_32_1230.png)


    0.9273367631576255
    


![png](output_32_1232.png)


    0.9021729315791581
    


![png](output_32_1234.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 38 	Training Loss: 0.170693 	Validation Loss: 0.376498
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9552780439481836
    


![png](output_32_1239.png)


    0.8943801861162473
    


![png](output_32_1241.png)


    0.9748505860783643
    


![png](output_32_1243.png)


    0.8344173155302885
    


![png](output_32_1245.png)


    0.9651037002138585
    


![png](output_32_1247.png)


    0.9596128763189387
    


![png](output_32_1249.png)


    0.9622355910571418
    


![png](output_32_1251.png)


    0.9381548525013741
    


![png](output_32_1253.png)


    0.9631219568676042
    


![png](output_32_1255.png)


    0.958378794351063
    


![png](output_32_1257.png)


    0.970290794118899
    


![png](output_32_1259.png)


    0.9759801863262434
    


![png](output_32_1261.png)


    0.9714279128062032
    


![png](output_32_1263.png)


    0.9503324679413909
    


![png](output_32_1265.png)


    0.9681796299305501
    


![png](output_32_1267.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 39 	Training Loss: 0.163081 	Validation Loss: 0.328027
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.929980905549266
    


![png](output_32_1272.png)


    0.9395226251252685
    


![png](output_32_1274.png)


    0.9706985291912827
    


![png](output_32_1276.png)


    0.9536004377095729
    


![png](output_32_1278.png)


    0.9655345347534019
    


![png](output_32_1280.png)


    0.9395378817985578
    


![png](output_32_1282.png)


    0.9314714338857473
    


![png](output_32_1284.png)


    0.9660401689140561
    


![png](output_32_1286.png)


    0.9618330409947775
    


![png](output_32_1288.png)


    0.8824624012478662
    


![png](output_32_1290.png)


    0.9669916422599462
    


![png](output_32_1292.png)


    0.9645120918348018
    


![png](output_32_1294.png)


    0.9741785485942616
    


![png](output_32_1296.png)


    0.9534227295531406
    


![png](output_32_1298.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 40 	Training Loss: 0.147729 	Validation Loss: 0.365852
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9544995585857903
    


![png](output_32_1303.png)


    0.9571441155325816
    


![png](output_32_1305.png)


    0.9771332470778991
    


![png](output_32_1307.png)


    0.9436676401155135
    


![png](output_32_1309.png)


    0.9669438311236292
    


![png](output_32_1311.png)


    0.9720892020652238
    


![png](output_32_1313.png)


    0.9468898726804895
    


![png](output_32_1315.png)


    0.9319940006027208
    


![png](output_32_1317.png)


    0.9566209574882916
    


![png](output_32_1319.png)


    0.969091908432174
    


![png](output_32_1321.png)


    0.9643111891490954
    


![png](output_32_1323.png)


    0.9632785346096838
    


![png](output_32_1325.png)


    0.9703137255987959
    


![png](output_32_1327.png)


    0.968680208419683
    


![png](output_32_1329.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 41 	Training Loss: 0.145410 	Validation Loss: 0.408220
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9046190170197326
    


![png](output_32_1334.png)


    0.9652121106718592
    


![png](output_32_1336.png)


    0.9676697427137105
    


![png](output_32_1338.png)


    0.9628237890042148
    


![png](output_32_1340.png)


    0.9265137865965389
    


![png](output_32_1342.png)


    0.9532768040289545
    


![png](output_32_1344.png)


    0.9673290476462378
    


![png](output_32_1346.png)


    0.9496419317135492
    


![png](output_32_1348.png)


    0.9595538822784196
    


![png](output_32_1350.png)


    0.9580980582945214
    


![png](output_32_1352.png)


    0.945762845034759
    


![png](output_32_1354.png)


    0.9626812864785622
    


![png](output_32_1356.png)


    0.9535445206688793
    


![png](output_32_1358.png)


    0.8658116629628367
    


![png](output_32_1360.png)


    0.9535220942920842
    


![png](output_32_1362.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 42 	Training Loss: 0.144185 	Validation Loss: 0.354790
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.964524539958439
    


![png](output_32_1367.png)


    0.9703191580846825
    


![png](output_32_1369.png)


    0.9389916468818965
    


![png](output_32_1371.png)


    0.9404724546238832
    


![png](output_32_1373.png)


    0.9702578654335992
    


![png](output_32_1375.png)


    0.9600054970833439
    


![png](output_32_1377.png)


    0.9631803568003885
    


![png](output_32_1379.png)


    0.9314086166532752
    


![png](output_32_1381.png)


    0.939242146887094
    


![png](output_32_1383.png)


    0.9700057761440917
    


![png](output_32_1385.png)


    0.7158808451212517
    


![png](output_32_1387.png)


    0.9685364985982468
    


![png](output_32_1389.png)


    0.9702772889772939
    


![png](output_32_1391.png)


    0.9589678440835518
    


![png](output_32_1393.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 43 	Training Loss: 0.138235 	Validation Loss: 0.353193
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9717450418749023
    


![png](output_32_1398.png)


    0.9684172044686049
    


![png](output_32_1400.png)


    0.9702172375484766
    


![png](output_32_1402.png)


    0.8796766556158204
    


![png](output_32_1404.png)


    0.9512113353491933
    


![png](output_32_1406.png)


    0.971072786766848
    


![png](output_32_1408.png)


    0.9759533474090913
    


![png](output_32_1410.png)


    0.9612778230154466
    


![png](output_32_1412.png)


    0.9276456097016106
    


![png](output_32_1414.png)


    0.9461748784155519
    


![png](output_32_1416.png)


    0.9694915126710009
    


![png](output_32_1418.png)


    0.9401626216153365
    


![png](output_32_1420.png)


    0.9629474440808057
    


![png](output_32_1422.png)


    0.9623279535276146
    


![png](output_32_1424.png)


    0.9624989843978106
    


![png](output_32_1426.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 44 	Training Loss: 0.143458 	Validation Loss: 0.339219
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9463697754439254
    


![png](output_32_1431.png)


    0.9506057314256553
    


![png](output_32_1433.png)


    0.9546364109495457
    


![png](output_32_1435.png)


    0.942055424761382
    


![png](output_32_1437.png)


    0.9710553231644208
    


![png](output_32_1439.png)


    0.9509787050458518
    


![png](output_32_1441.png)


    0.9403030804344912
    


![png](output_32_1443.png)


    0.9675679455089605
    


![png](output_32_1445.png)


    0.9603147608938984
    


![png](output_32_1447.png)


    0.9562938242530006
    


![png](output_32_1449.png)


    0.9684479696105283
    


![png](output_32_1451.png)


    0.9662514717438194
    


![png](output_32_1453.png)


    0.959045898211875
    


![png](output_32_1455.png)


    0.9785203938452806
    


![png](output_32_1457.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 45 	Training Loss: 0.136481 	Validation Loss: 0.379480
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9377299326415094
    


![png](output_32_1462.png)


    0.970785650714237
    


![png](output_32_1464.png)


    0.9683191377415189
    


![png](output_32_1466.png)


    0.9703507610326632
    


![png](output_32_1468.png)


    0.9360526609881542
    


![png](output_32_1470.png)


    0.9235928655101415
    


![png](output_32_1472.png)


    0.9628398127035094
    


![png](output_32_1474.png)


    0.9509132337814095
    


![png](output_32_1476.png)


    0.9205185350233445
    


![png](output_32_1478.png)


    0.9739391069023229
    


![png](output_32_1480.png)


    0.9349238130880155
    


![png](output_32_1482.png)


    0.9578061022487614
    


![png](output_32_1484.png)


    0.9660047688620508
    


![png](output_32_1486.png)


    0.9444022799279834
    


![png](output_32_1488.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 46 	Training Loss: 0.136690 	Validation Loss: 0.406464
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.945129283872326
    


![png](output_32_1493.png)


    0.960909486871441
    


![png](output_32_1495.png)


    0.8928088164575099
    


![png](output_32_1497.png)


    0.9753414640322984
    


![png](output_32_1499.png)


    0.9622793388538189
    


![png](output_32_1501.png)


    0.9690110656199455
    


![png](output_32_1503.png)


    0.9707346280207995
    


![png](output_32_1505.png)


    0.9614553290219787
    


![png](output_32_1507.png)


    0.9350935774505349
    


![png](output_32_1509.png)


    0.970282739188187
    


![png](output_32_1511.png)


    0.9650826200293441
    


![png](output_32_1513.png)


    0.9738009887065289
    


![png](output_32_1515.png)


    0.9559872828834134
    


![png](output_32_1517.png)


    0.9625034512080288
    


![png](output_32_1519.png)


    0.9678405031988936
    


![png](output_32_1521.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 47 	Training Loss: 0.144153 	Validation Loss: 0.354173
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9613589779614429
    


![png](output_32_1526.png)


    0.9653073830572232
    


![png](output_32_1528.png)


    0.9641239822213306
    


![png](output_32_1530.png)


    0.9644350264801111
    


![png](output_32_1532.png)


    0.95175559562823
    


![png](output_32_1534.png)


    0.9618147664950145
    


![png](output_32_1536.png)


    0.9644942963692406
    


![png](output_32_1538.png)


    0.9544285804196377
    


![png](output_32_1540.png)


    0.9484501236772303
    


![png](output_32_1542.png)


    0.9527375625213774
    


![png](output_32_1544.png)


    0.9496950075724828
    


![png](output_32_1546.png)


    0.9806658857128701
    


![png](output_32_1548.png)


    0.9348830588534544
    


![png](output_32_1550.png)


    0.9580281841318763
    


![png](output_32_1552.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 48 	Training Loss: 0.145921 	Validation Loss: 0.379303
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9735785987866116
    


![png](output_32_1557.png)


    0.962682453688926
    


![png](output_32_1559.png)


    0.9671101494140423
    


![png](output_32_1561.png)


    0.9584546193489748
    


![png](output_32_1563.png)


    0.9618393724034743
    


![png](output_32_1565.png)


    0.9553452550665199
    


![png](output_32_1567.png)


    0.9621209047541001
    


![png](output_32_1569.png)


    0.9585002658495332
    


![png](output_32_1571.png)


    0.9634211090082736
    


![png](output_32_1573.png)


    0.9725873725251721
    


![png](output_32_1575.png)


    0.9128799156157796
    


![png](output_32_1577.png)


    0.9656413343859656
    


![png](output_32_1579.png)


    0.9543752300257895
    


![png](output_32_1581.png)


    0.96994299672434
    


![png](output_32_1583.png)


    0.9776120785160677
    


![png](output_32_1585.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 49 	Training Loss: 0.131413 	Validation Loss: 0.353326
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9772951016398734
    


![png](output_32_1590.png)


    0.9755025706591476
    


![png](output_32_1592.png)


    0.9681609743238455
    


![png](output_32_1594.png)


    0.9538078479394577
    


![png](output_32_1596.png)


    0.9670355383139625
    


![png](output_32_1598.png)


    0.963381554352708
    


![png](output_32_1600.png)


    0.947628277317369
    


![png](output_32_1602.png)


    0.9519725944381211
    


![png](output_32_1604.png)


    0.9579662761794412
    


![png](output_32_1606.png)


    0.94926545160209
    


![png](output_32_1608.png)


    0.9713799160340404
    


![png](output_32_1610.png)


    0.9640665367562857
    


![png](output_32_1612.png)


    0.9591781441905669
    


![png](output_32_1614.png)


    0.8765368453335414
    


![png](output_32_1616.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 50 	Training Loss: 0.135846 	Validation Loss: 0.349992
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9516863571807261
    


![png](output_32_1621.png)


    0.9505228005432693
    


![png](output_32_1623.png)


    0.9659354738781583
    


![png](output_32_1625.png)


    0.9580918005853823
    


![png](output_32_1627.png)


    0.9639423738040049
    


![png](output_32_1629.png)


    0.9547862984664329
    


![png](output_32_1631.png)


    0.9624686182839881
    


![png](output_32_1633.png)


    0.960716158473808
    


![png](output_32_1635.png)


    0.9570732611822057
    


![png](output_32_1637.png)


    0.9550981175132456
    


![png](output_32_1639.png)


    0.969128663042287
    


![png](output_32_1641.png)


    0.9561828483809671
    


![png](output_32_1643.png)


    0.9515721329265687
    


![png](output_32_1645.png)


    0.9625294556862043
    


![png](output_32_1647.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 51 	Training Loss: 0.137179 	Validation Loss: 0.431712
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9072985208231776
    


![png](output_32_1652.png)


    0.95614547096272
    


![png](output_32_1654.png)


    0.9723059312210718
    


![png](output_32_1656.png)


    0.9623329439291279
    


![png](output_32_1658.png)


    0.965464489617741
    


![png](output_32_1660.png)


    0.961419039836365
    


![png](output_32_1662.png)


    0.9619508037037721
    


![png](output_32_1664.png)


    0.9722755600830407
    


![png](output_32_1666.png)


    0.966471702617656
    


![png](output_32_1668.png)


    0.9626198334531606
    


![png](output_32_1670.png)


    0.967407822678757
    


![png](output_32_1672.png)


    0.9674263030027745
    


![png](output_32_1674.png)


    0.9613963411660418
    


![png](output_32_1676.png)


    0.9674058206144303
    


![png](output_32_1678.png)


    0.9452017741901615
    


![png](output_32_1680.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 52 	Training Loss: 0.138401 	Validation Loss: 0.381881
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9514012851202474
    


![png](output_32_1685.png)


    0.9620032821626272
    


![png](output_32_1687.png)


    0.9638900010217866
    


![png](output_32_1689.png)


    0.9734169882056468
    


![png](output_32_1691.png)


    0.9637285717230872
    


![png](output_32_1693.png)


    0.9793859835107044
    


![png](output_32_1695.png)


    0.9574885742373471
    


![png](output_32_1697.png)


    0.9703764585415035
    


![png](output_32_1699.png)


    0.9645588219465033
    


![png](output_32_1701.png)


    0.9597494130792525
    


![png](output_32_1703.png)


    0.9665263076620279
    


![png](output_32_1705.png)


    0.9757370690703702
    


![png](output_32_1707.png)


    0.9716886272279974
    


![png](output_32_1709.png)


    0.9234067426463202
    


![png](output_32_1711.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 53 	Training Loss: 0.134306 	Validation Loss: 0.389960
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.94620650853326
    


![png](output_32_1716.png)


    0.9742400505949369
    


![png](output_32_1718.png)


    0.9614551534827795
    


![png](output_32_1720.png)


    0.9659160535478819
    


![png](output_32_1722.png)


    0.9583968405870485
    


![png](output_32_1724.png)


    0.9760286868032881
    


![png](output_32_1726.png)


    0.9752219838878662
    


![png](output_32_1728.png)


    0.9510521153377003
    


![png](output_32_1730.png)


    0.9566624267089286
    


![png](output_32_1732.png)


    0.9730551320464338
    


![png](output_32_1734.png)


    0.9611147107276223
    


![png](output_32_1736.png)


    0.9654399441817214
    


![png](output_32_1738.png)


    0.9626929156038765
    


![png](output_32_1740.png)


    0.9781638745514056
    


![png](output_32_1742.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 54 	Training Loss: 0.119321 	Validation Loss: 0.367791
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.959808162618621
    


![png](output_32_1747.png)


    0.974835787995707
    


![png](output_32_1749.png)


    0.9532056782869993
    


![png](output_32_1751.png)


    0.9620545113856434
    


![png](output_32_1753.png)


    0.9641543756525159
    


![png](output_32_1755.png)


    0.9643330436217485
    


![png](output_32_1757.png)


    0.9627810632180739
    


![png](output_32_1759.png)


    0.9659318611186267
    


![png](output_32_1761.png)


    0.9473541878629766
    


![png](output_32_1763.png)


    0.9746909799546379
    


![png](output_32_1765.png)


    0.9644679820620585
    


![png](output_32_1767.png)


    0.963421309835622
    


![png](output_32_1769.png)


    0.9603977071519855
    


![png](output_32_1771.png)


    0.961834936314095
    


![png](output_32_1773.png)


    0.9621351455051763
    


![png](output_32_1775.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 55 	Training Loss: 0.131015 	Validation Loss: 0.401339
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.945087419887463
    


![png](output_32_1780.png)


    0.8757877053254315
    


![png](output_32_1782.png)


    0.9621825411806595
    


![png](output_32_1784.png)


    0.9558830136281238
    


![png](output_32_1786.png)


    0.9645620551642715
    


![png](output_32_1788.png)


    0.9690171843135191
    


![png](output_32_1790.png)


    0.9727711053255317
    


![png](output_32_1792.png)


    0.9578419978557058
    


![png](output_32_1794.png)


    0.9001339776859634
    


![png](output_32_1796.png)


    0.9680719206077915
    


![png](output_32_1798.png)


    0.9772631093402613
    


![png](output_32_1800.png)


    0.9706943543415367
    


![png](output_32_1802.png)


    0.9598221940644273
    


![png](output_32_1804.png)


    0.9414986478269995
    


![png](output_32_1806.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 56 	Training Loss: 0.142573 	Validation Loss: 0.406023
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9776070012565612
    


![png](output_32_1811.png)


    0.9632785428437931
    


![png](output_32_1813.png)


    0.9617597417384054
    


![png](output_32_1815.png)


    0.8676846106024965
    


![png](output_32_1817.png)


    0.9218292443161529
    


![png](output_32_1819.png)


    0.9710549323762839
    


![png](output_32_1821.png)


    0.9575391951299526
    


![png](output_32_1823.png)


    0.9620486170066772
    


![png](output_32_1825.png)


    0.9630547678342523
    


![png](output_32_1827.png)


    0.96374945968617
    


![png](output_32_1829.png)


    0.9589115031390945
    


![png](output_32_1831.png)


    0.9607609590630778
    


![png](output_32_1833.png)


    0.97395506053131
    


![png](output_32_1835.png)


    0.9673496453034156
    


![png](output_32_1837.png)


    0.8934973270793193
    


![png](output_32_1839.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 57 	Training Loss: 0.121519 	Validation Loss: 0.403714
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9732391527864129
    


![png](output_32_1844.png)


    0.9639888659187756
    


![png](output_32_1846.png)


    0.9707375329081867
    


![png](output_32_1848.png)


    0.9332026524019474
    


![png](output_32_1850.png)


    0.967895085392572
    


![png](output_32_1852.png)


    0.9572323585777238
    


![png](output_32_1854.png)


    0.960338482242544
    


![png](output_32_1856.png)


    0.9450667321146755
    


![png](output_32_1858.png)


    0.9682710414702091
    


![png](output_32_1860.png)


    0.9550239710897621
    


![png](output_32_1862.png)


    0.9722999428319481
    


![png](output_32_1864.png)


    0.9471302250820741
    


![png](output_32_1866.png)


    0.9405775615365908
    


![png](output_32_1868.png)


    0.8677968836502133
    


![png](output_32_1870.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 58 	Training Loss: 0.134013 	Validation Loss: 0.403297
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9559729243190089
    


![png](output_32_1875.png)


    0.9606293416450631
    


![png](output_32_1877.png)


    0.9602585698353059
    


![png](output_32_1879.png)


    0.9699073167935893
    


![png](output_32_1881.png)


    0.9743385199065577
    


![png](output_32_1883.png)


    0.9785436047267142
    


![png](output_32_1885.png)


    0.9727463864420782
    


![png](output_32_1887.png)


    0.9653409868623177
    


![png](output_32_1889.png)


    0.9632311861494374
    


![png](output_32_1891.png)


    0.964209626850066
    


![png](output_32_1893.png)


    0.9627431550163743
    


![png](output_32_1895.png)


    0.9707709459728541
    


![png](output_32_1897.png)


    0.9630715433643786
    


![png](output_32_1899.png)


    0.9604195032345729
    


![png](output_32_1901.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 59 	Training Loss: 0.128329 	Validation Loss: 0.382462
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9772633753104343
    


![png](output_32_1906.png)


    0.9738724799658272
    


![png](output_32_1908.png)


    0.9687449964036154
    


![png](output_32_1910.png)


    0.9688960714056721
    


![png](output_32_1912.png)


    0.9697458648607109
    


![png](output_32_1914.png)


    0.9653001389899483
    


![png](output_32_1916.png)


    0.9609618638236217
    


![png](output_32_1918.png)


    0.9352932144624195
    


![png](output_32_1920.png)


    0.9762531410046962
    


![png](output_32_1922.png)


    0.9732552009572848
    


![png](output_32_1924.png)


    0.9709746351692367
    


![png](output_32_1926.png)


    0.9734982302364116
    


![png](output_32_1928.png)


    0.9683893702866155
    


![png](output_32_1930.png)


    0.9536821130099433
    


![png](output_32_1932.png)


    0.9442695869153273
    


![png](output_32_1934.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 60 	Training Loss: 0.121381 	Validation Loss: 0.377088
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9737997318228934
    


![png](output_32_1939.png)


    0.962726504019938
    


![png](output_32_1941.png)


    0.9738463034814777
    


![png](output_32_1943.png)


    0.9743067046561302
    


![png](output_32_1945.png)


    0.9698879808450396
    


![png](output_32_1947.png)


    0.9636647987262329
    


![png](output_32_1949.png)


    0.9520719474183295
    


![png](output_32_1951.png)


    0.9755269616892961
    


![png](output_32_1953.png)


    0.9778772764781688
    


![png](output_32_1955.png)


    0.9633455305339985
    


![png](output_32_1957.png)


    0.9556601109116871
    


![png](output_32_1959.png)


    0.9589585373138427
    


![png](output_32_1961.png)


    0.9726027589778561
    


![png](output_32_1963.png)


    0.971608080700783
    


![png](output_32_1965.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 61 	Training Loss: 0.129896 	Validation Loss: 0.385751
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9444282294832647
    


![png](output_32_1970.png)


    0.972295657276576
    


![png](output_32_1972.png)


    0.9703021623477157
    


![png](output_32_1974.png)


    0.9611447352757725
    


![png](output_32_1976.png)


    0.9722563399699455
    


![png](output_32_1978.png)


    0.9057232342742878
    


![png](output_32_1980.png)


    0.951060571983753
    


![png](output_32_1982.png)


    0.9353142386625335
    


![png](output_32_1984.png)


    0.9677460740950358
    


![png](output_32_1986.png)


    0.8629273811810303
    


![png](output_32_1988.png)


    0.9571758887116297
    


![png](output_32_1990.png)


    0.9749726371382156
    


![png](output_32_1992.png)


    0.9391122662501147
    


![png](output_32_1994.png)


    0.9632270836133703
    


![png](output_32_1996.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 62 	Training Loss: 0.129312 	Validation Loss: 0.381404
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9675601258732145
    


![png](output_32_2001.png)


    0.9707686364141996
    


![png](output_32_2003.png)


    0.9588499054240747
    


![png](output_32_2005.png)


    0.9563647849065067
    


![png](output_32_2007.png)


    0.9590854958828896
    


![png](output_32_2009.png)


    0.9686997946041709
    


![png](output_32_2011.png)


    0.974255790376376
    


![png](output_32_2013.png)


    0.96414745716683
    


![png](output_32_2015.png)


    0.9644272610320992
    


![png](output_32_2017.png)


    0.9582848637204017
    


![png](output_32_2019.png)


    0.9725396688315229
    


![png](output_32_2021.png)


    0.9659394096408906
    


![png](output_32_2023.png)


    0.9483594277616941
    


![png](output_32_2025.png)


    0.9607751941381464
    


![png](output_32_2027.png)


    0.9641107244737654
    


![png](output_32_2029.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 63 	Training Loss: 0.121189 	Validation Loss: 0.405157
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.974113200377658
    


![png](output_32_2034.png)


    0.9605166905994429
    


![png](output_32_2036.png)


    0.962950137555288
    


![png](output_32_2038.png)


    0.9776264188385155
    


![png](output_32_2040.png)


    0.9701481246070163
    


![png](output_32_2042.png)


    0.9727408866387404
    


![png](output_32_2044.png)


    0.9729275840596211
    


![png](output_32_2046.png)


    0.9562324711784739
    


![png](output_32_2048.png)


    0.9642454980273806
    


![png](output_32_2050.png)


    0.9703165315778005
    


![png](output_32_2052.png)


    0.9649925438964676
    


![png](output_32_2054.png)


    0.971546961834589
    


![png](output_32_2056.png)


    0.9668639419097778
    


![png](output_32_2058.png)


    0.9674666279127183
    


![png](output_32_2060.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 64 	Training Loss: 0.128859 	Validation Loss: 0.358684
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9577974937257225
    


![png](output_32_2065.png)


    0.8558558440850317
    


![png](output_32_2067.png)


    0.96819274773321
    


![png](output_32_2069.png)


    0.9711105037735992
    


![png](output_32_2071.png)


    0.9635725609623433
    


![png](output_32_2073.png)


    0.9660697956359376
    


![png](output_32_2075.png)


    0.9612530802990544
    


![png](output_32_2077.png)


    0.9537497352240543
    


![png](output_32_2079.png)


    0.9624171171012748
    


![png](output_32_2081.png)


    0.9283905174317961
    


![png](output_32_2083.png)


    0.9586714957377335
    


![png](output_32_2085.png)


    0.9356593116905015
    


![png](output_32_2087.png)


    0.9636263366812374
    


![png](output_32_2089.png)


    0.9433131110326585
    


![png](output_32_2091.png)


    0.9670873191276984
    


![png](output_32_2093.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 65 	Training Loss: 0.115428 	Validation Loss: 0.365261
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.954681134697521
    


![png](output_32_2098.png)


    0.9770110175736216
    


![png](output_32_2100.png)


    0.9543579206737121
    


![png](output_32_2102.png)


    0.9723240811746138
    


![png](output_32_2104.png)


    0.9716368285518003
    


![png](output_32_2106.png)


    0.9696492318747064
    


![png](output_32_2108.png)


    0.9641196887500902
    


![png](output_32_2110.png)


    0.9849587804400861
    


![png](output_32_2112.png)


    0.9659785133329152
    


![png](output_32_2114.png)


    0.9777318607220997
    


![png](output_32_2116.png)


    0.9657685042355139
    


![png](output_32_2118.png)


    0.9695100744048277
    


![png](output_32_2120.png)


    0.9718806078828688
    


![png](output_32_2122.png)


    0.9749317467500398
    


![png](output_32_2124.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 66 	Training Loss: 0.119124 	Validation Loss: 0.354104
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.960090802103958
    


![png](output_32_2129.png)


    0.9592526170092327
    


![png](output_32_2131.png)


    0.9699462229459518
    


![png](output_32_2133.png)


    0.9741363884351797
    


![png](output_32_2135.png)


    0.9767358476738592
    


![png](output_32_2137.png)


    0.9661542979033437
    


![png](output_32_2139.png)


    0.9499418106671746
    


![png](output_32_2141.png)


    0.9618366417861917
    


![png](output_32_2143.png)


    0.9654312787122046
    


![png](output_32_2145.png)


    0.9588128795152541
    


![png](output_32_2147.png)


    0.974191391858381
    


![png](output_32_2149.png)


    0.9706850118344859
    


![png](output_32_2151.png)


    0.9662078048356262
    


![png](output_32_2153.png)


    0.9665494142531842
    


![png](output_32_2155.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 67 	Training Loss: 0.109443 	Validation Loss: 0.422344
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9719662952393391
    


![png](output_32_2160.png)


    0.9648349239844247
    


![png](output_32_2162.png)


    0.9411021165477521
    


![png](output_32_2164.png)


    0.9760978797235547
    


![png](output_32_2166.png)


    0.9763836233243159
    


![png](output_32_2168.png)


    0.9795567993739456
    


![png](output_32_2170.png)


    0.9612820546696035
    


![png](output_32_2172.png)


    0.9709486123070786
    


![png](output_32_2174.png)


    0.9533048753889162
    


![png](output_32_2176.png)


    0.9640835945637766
    


![png](output_32_2178.png)


    0.967854732002716
    


![png](output_32_2180.png)


    0.9560035037340315
    


![png](output_32_2182.png)


    0.9680896643773707
    


![png](output_32_2184.png)


    0.9686947391166076
    


![png](output_32_2186.png)


    0.9661395112471465
    


![png](output_32_2188.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 68 	Training Loss: 0.122707 	Validation Loss: 0.402819
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9649485989932848
    


![png](output_32_2193.png)


    0.9667439940205891
    


![png](output_32_2195.png)


    0.965117587167269
    


![png](output_32_2197.png)


    0.9436095726920104
    


![png](output_32_2199.png)


    0.9685488603040466
    


![png](output_32_2201.png)


    0.9700650202569974
    


![png](output_32_2203.png)


    0.9605829900602546
    


![png](output_32_2205.png)


    0.9693652639775274
    


![png](output_32_2207.png)


    0.9722945128407079
    


![png](output_32_2209.png)


    0.9764881358405442
    


![png](output_32_2211.png)


    0.9669863266702141
    


![png](output_32_2213.png)


    0.9437048183251913
    


![png](output_32_2215.png)


    0.970123646923074
    


![png](output_32_2217.png)


    0.9627514003864368
    


![png](output_32_2219.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 69 	Training Loss: 0.120838 	Validation Loss: 0.384049
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9414590528294935
    


![png](output_32_2224.png)


    0.9664623604228761
    


![png](output_32_2226.png)


    0.9633249366226788
    


![png](output_32_2228.png)


    0.9806951431813553
    


![png](output_32_2230.png)


    0.9437845418403391
    


![png](output_32_2232.png)


    0.9605659000177905
    


![png](output_32_2234.png)


    0.9314169485121206
    


![png](output_32_2236.png)


    0.9638097510169745
    


![png](output_32_2238.png)


    0.9505549869675104
    


![png](output_32_2240.png)


    0.979271165569432
    


![png](output_32_2242.png)


    0.9309408347430395
    


![png](output_32_2244.png)


    0.9704037276695889
    


![png](output_32_2246.png)


    0.9585266971969202
    


![png](output_32_2248.png)


    0.9704551422352201
    


![png](output_32_2250.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 70 	Training Loss: 0.134337 	Validation Loss: 0.362589
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9707180791524821
    


![png](output_32_2255.png)


    0.9695632971737982
    


![png](output_32_2257.png)


    0.9716105186443951
    


![png](output_32_2259.png)


    0.9641584789304635
    


![png](output_32_2261.png)


    0.965439918397184
    


![png](output_32_2263.png)


    0.9653382365540752
    


![png](output_32_2265.png)


    0.9437859530161927
    


![png](output_32_2267.png)


    0.8326464384527518
    


![png](output_32_2269.png)


    0.9720711232596845
    


![png](output_32_2271.png)


    0.9655954496993089
    


![png](output_32_2273.png)


    0.9736981184422819
    


![png](output_32_2275.png)


    0.9651470644285841
    


![png](output_32_2277.png)


    0.9722478970904915
    


![png](output_32_2279.png)


    0.9737136569779926
    


![png](output_32_2281.png)


    0.9788267900216312
    


![png](output_32_2283.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 71 	Training Loss: 0.129580 	Validation Loss: 0.387579
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9600421427319983
    


![png](output_32_2288.png)


    0.9710233117218836
    


![png](output_32_2290.png)


    0.9668719344545527
    


![png](output_32_2292.png)


    0.9659226687364877
    


![png](output_32_2294.png)


    0.9617655885132098
    


![png](output_32_2296.png)


    0.9550330964593836
    


![png](output_32_2298.png)


    0.9744339165887973
    


![png](output_32_2300.png)


    0.969456529013591
    


![png](output_32_2302.png)


    0.9561378931895731
    


![png](output_32_2304.png)


    0.9644707922677427
    


![png](output_32_2306.png)


    0.9806641347052382
    


![png](output_32_2308.png)


    0.9670975797553778
    


![png](output_32_2310.png)


    0.9428447562492459
    


![png](output_32_2312.png)


    0.9706296760843658
    


![png](output_32_2314.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 72 	Training Loss: 0.113837 	Validation Loss: 0.346054
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9673706162414131
    


![png](output_32_2319.png)


    0.9723171449172079
    


![png](output_32_2321.png)


    0.9661006813334159
    


![png](output_32_2323.png)


    0.9687660808041898
    


![png](output_32_2325.png)


    0.9678044657321514
    


![png](output_32_2327.png)


    0.9709919189341634
    


![png](output_32_2329.png)


    0.9551389403914419
    


![png](output_32_2331.png)


    0.9729442737122425
    


![png](output_32_2333.png)


    0.956965178578261
    


![png](output_32_2335.png)


    0.9764036754171483
    


![png](output_32_2337.png)


    0.9590674674862265
    


![png](output_32_2339.png)


    0.9516154398843368
    


![png](output_32_2341.png)


    0.972514176989848
    


![png](output_32_2343.png)


    0.9685690271675509
    


![png](output_32_2345.png)


    0.9678928312176986
    


![png](output_32_2347.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 73 	Training Loss: 0.118824 	Validation Loss: 0.381945
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9806247722216587
    


![png](output_32_2352.png)


    0.9550886137956897
    


![png](output_32_2354.png)


    0.9769441067182016
    


![png](output_32_2356.png)


    0.9660775493536155
    


![png](output_32_2358.png)


    0.9616768156734771
    


![png](output_32_2360.png)


    0.9522954394727698
    


![png](output_32_2362.png)


    0.9535260095536443
    


![png](output_32_2364.png)


    0.9558945396232125
    


![png](output_32_2366.png)


    0.9763267558694635
    


![png](output_32_2368.png)


    0.9059096248764097
    


![png](output_32_2370.png)


    0.968940298394257
    


![png](output_32_2372.png)


    0.9541299587511735
    


![png](output_32_2374.png)


    0.969324736954926
    


![png](output_32_2376.png)


    0.9562186784816913
    


![png](output_32_2378.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 74 	Training Loss: 0.121195 	Validation Loss: 0.387399
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9669123418362128
    


![png](output_32_2383.png)


    0.9715260466343005
    


![png](output_32_2385.png)


    0.9672957673979217
    


![png](output_32_2387.png)


    0.9644947822777469
    


![png](output_32_2389.png)


    0.9713067097891319
    


![png](output_32_2391.png)


    0.9683650277295086
    


![png](output_32_2393.png)


    0.9739623405398097
    


![png](output_32_2395.png)


    0.9745891572603227
    


![png](output_32_2397.png)


    0.9739796062458141
    


![png](output_32_2399.png)


    0.9767649211045044
    


![png](output_32_2401.png)


    0.9588474780391922
    


![png](output_32_2403.png)


    0.9764210818150121
    


![png](output_32_2405.png)


    0.9722651910275679
    


![png](output_32_2407.png)


    0.9571287306661245
    


![png](output_32_2409.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 75 	Training Loss: 0.112058 	Validation Loss: 0.394343
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9775702585673065
    


![png](output_32_2414.png)


    0.9678676258239277
    


![png](output_32_2416.png)


    0.9728894290620098
    


![png](output_32_2418.png)


    0.946906403662533
    


![png](output_32_2420.png)


    0.9733819488881281
    


![png](output_32_2422.png)


    0.9698801867647
    


![png](output_32_2424.png)


    0.9567482649648307
    


![png](output_32_2426.png)


    0.979133478051063
    


![png](output_32_2428.png)


    0.9736784666405238
    


![png](output_32_2430.png)


    0.9686891225301516
    


![png](output_32_2432.png)


    0.9796047104306159
    


![png](output_32_2434.png)


    0.9703341712112978
    


![png](output_32_2436.png)


    0.9727515554056024
    


![png](output_32_2438.png)


    0.9727182229877496
    


![png](output_32_2440.png)


    0.9574581421125976
    


![png](output_32_2442.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 76 	Training Loss: 0.104025 	Validation Loss: 0.322641
    Validation loss decreased (0.323970 --> 0.322641).  Saving model ...
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9630932108489993
    


![png](output_32_2447.png)


    0.9620084691945727
    


![png](output_32_2449.png)


    0.930772968332884
    


![png](output_32_2451.png)


    0.960249865102083
    


![png](output_32_2453.png)


    0.9829418892893824
    


![png](output_32_2455.png)


    0.9782590557322668
    


![png](output_32_2457.png)


    0.971922720733808
    


![png](output_32_2459.png)


    0.9728518664721919
    


![png](output_32_2461.png)


    0.9744056802135111
    


![png](output_32_2463.png)


    0.9661634460384327
    


![png](output_32_2465.png)


    0.9771362180763483
    


![png](output_32_2467.png)


    0.9676659167903126
    


![png](output_32_2469.png)


    0.9616393858272644
    


![png](output_32_2471.png)


    0.9761816228707121
    


![png](output_32_2473.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 77 	Training Loss: 0.107173 	Validation Loss: 0.409794
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.975443262537325
    


![png](output_32_2478.png)


    0.9820235794118359
    


![png](output_32_2480.png)


    0.9725499379170932
    


![png](output_32_2482.png)


    0.9651917126661788
    


![png](output_32_2484.png)


    0.9755544025504279
    


![png](output_32_2486.png)


    0.9694933132955663
    


![png](output_32_2488.png)


    0.970492544534403
    


![png](output_32_2490.png)


    0.9625612315052735
    


![png](output_32_2492.png)


    0.9750930219018359
    


![png](output_32_2494.png)


    0.9820297842142918
    


![png](output_32_2496.png)


    0.9758672800879145
    


![png](output_32_2498.png)


    0.979201907619085
    


![png](output_32_2500.png)


    0.9595978553674303
    


![png](output_32_2502.png)


    0.9791824813819643
    


![png](output_32_2504.png)


    0.9780813614416051
    


![png](output_32_2506.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 78 	Training Loss: 0.110003 	Validation Loss: 0.360645
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9693344684443145
    


![png](output_32_2511.png)


    0.9626157444789954
    


![png](output_32_2513.png)


    0.9657667137404942
    


![png](output_32_2515.png)


    0.9744087824184509
    


![png](output_32_2517.png)


    0.9746481417032257
    


![png](output_32_2519.png)


    0.969349972673014
    


![png](output_32_2521.png)


    0.9623720211445048
    


![png](output_32_2523.png)


    0.9702293020558772
    


![png](output_32_2525.png)


    0.9723775995538382
    


![png](output_32_2527.png)


    0.9791419934257449
    


![png](output_32_2529.png)


    0.9713832552338048
    


![png](output_32_2531.png)


    0.9756327519574302
    


![png](output_32_2533.png)


    0.973555597597308
    


![png](output_32_2535.png)


    0.9777806638662769
    


![png](output_32_2537.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 79 	Training Loss: 0.106256 	Validation Loss: 0.356625
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9747595270157653
    


![png](output_32_2542.png)


    0.978603798798072
    


![png](output_32_2544.png)


    0.971604914512277
    


![png](output_32_2546.png)


    0.9555509466262375
    


![png](output_32_2548.png)


    0.9570040518562242
    


![png](output_32_2550.png)


    0.9628997735241105
    


![png](output_32_2552.png)


    0.9492470959522523
    


![png](output_32_2554.png)


    0.9785109277603024
    


![png](output_32_2556.png)


    0.9722966327774283
    


![png](output_32_2558.png)


    0.93331146749398
    


![png](output_32_2560.png)


    0.9694163207807098
    


![png](output_32_2562.png)


    0.9630142570659226
    


![png](output_32_2564.png)


    0.961601172811881
    


![png](output_32_2566.png)


    0.9579396004268469
    


![png](output_32_2568.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 80 	Training Loss: 0.110195 	Validation Loss: 0.412041
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9775784697337243
    


![png](output_32_2573.png)


    0.975887230941547
    


![png](output_32_2575.png)


    0.9769461760195807
    


![png](output_32_2577.png)


    0.9708078599435275
    


![png](output_32_2579.png)


    0.9738087351101419
    


![png](output_32_2581.png)


    0.9627431015606822
    


![png](output_32_2583.png)


    0.9691111669293191
    


![png](output_32_2585.png)


    0.9695818412615933
    


![png](output_32_2587.png)


    0.9662892604166216
    


![png](output_32_2589.png)


    0.9603355655751136
    


![png](output_32_2591.png)


    0.9701183058905121
    


![png](output_32_2593.png)


    0.9739119580397195
    


![png](output_32_2595.png)


    0.9714969444659698
    


![png](output_32_2597.png)


    0.9707176882633283
    


![png](output_32_2599.png)


    0.9596420834017254
    


![png](output_32_2601.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 81 	Training Loss: 0.108721 	Validation Loss: 0.348041
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9737460527193902
    


![png](output_32_2606.png)


    0.9751017103651927
    


![png](output_32_2608.png)


    0.9700799148764978
    


![png](output_32_2610.png)


    0.9719643607218054
    


![png](output_32_2612.png)


    0.9651589258264092
    


![png](output_32_2614.png)


    0.973780261256734
    


![png](output_32_2616.png)


    0.9636798193236172
    


![png](output_32_2618.png)


    0.9748293622462079
    


![png](output_32_2620.png)


    0.9647217682407809
    


![png](output_32_2622.png)


    0.9692398761425871
    


![png](output_32_2624.png)


    0.94866282490558
    


![png](output_32_2626.png)


    0.953384789422264
    


![png](output_32_2628.png)


    0.9606976410896892
    


![png](output_32_2630.png)


    0.966710817606463
    


![png](output_32_2632.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 82 	Training Loss: 0.113025 	Validation Loss: 0.378905
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9703963596168332
    


![png](output_32_2637.png)


    0.9575325331948115
    


![png](output_32_2639.png)


    0.9551422583846116
    


![png](output_32_2641.png)


    0.9641426454646773
    


![png](output_32_2643.png)


    0.9694550611852242
    


![png](output_32_2645.png)


    0.9683107478428217
    


![png](output_32_2647.png)


    0.9713016333182669
    


![png](output_32_2649.png)


    0.9679120627065659
    


![png](output_32_2651.png)


    0.9605639031756821
    


![png](output_32_2653.png)


    0.9769648840105691
    


![png](output_32_2655.png)


    0.9717159045485684
    


![png](output_32_2657.png)


    0.9745565743820911
    


![png](output_32_2659.png)


    0.9702630495536919
    


![png](output_32_2661.png)


    0.9769095229601223
    


![png](output_32_2663.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 83 	Training Loss: 0.103454 	Validation Loss: 0.360910
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.974693450080675
    


![png](output_32_2668.png)


    0.9769826208302896
    


![png](output_32_2670.png)


    0.9728157487733347
    


![png](output_32_2672.png)


    0.9783393635493838
    


![png](output_32_2674.png)


    0.9593781761835345
    


![png](output_32_2676.png)


    0.9619819662694326
    


![png](output_32_2678.png)


    0.9717161611592426
    


![png](output_32_2680.png)


    0.9643958835414042
    


![png](output_32_2682.png)


    0.975024975800602
    


![png](output_32_2684.png)


    0.967445729114387
    


![png](output_32_2686.png)


    0.964859488096248
    


![png](output_32_2688.png)


    0.955340830611385
    


![png](output_32_2690.png)


    0.9737979178423836
    


![png](output_32_2692.png)


    0.9679494231742918
    


![png](output_32_2694.png)


    0.9776579475602958
    


![png](output_32_2696.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 84 	Training Loss: 0.109970 	Validation Loss: 0.374516
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9628837184736604
    


![png](output_32_2701.png)


    0.9809797349307039
    


![png](output_32_2703.png)


    0.9696769391784061
    


![png](output_32_2705.png)


    0.9760956349112154
    


![png](output_32_2707.png)


    0.9749319987852814
    


![png](output_32_2709.png)


    0.9721408027139247
    


![png](output_32_2711.png)


    0.9712761563219272
    


![png](output_32_2713.png)


    0.9693503813312384
    


![png](output_32_2715.png)


    0.9719740310284473
    


![png](output_32_2717.png)


    0.9744458489720538
    


![png](output_32_2719.png)


    0.9739599964600111
    


![png](output_32_2721.png)


    0.9487891064925025
    


![png](output_32_2723.png)


    0.9257058557194269
    


![png](output_32_2725.png)


    0.9687837217030406
    


![png](output_32_2727.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 85 	Training Loss: 0.116178 	Validation Loss: 0.394166
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9807182326287888
    


![png](output_32_2732.png)


    0.9778982602005886
    


![png](output_32_2734.png)


    0.964974773225491
    


![png](output_32_2736.png)


    0.9641255452294974
    


![png](output_32_2738.png)


    0.9571298595773264
    


![png](output_32_2740.png)


    0.9657001871843015
    


![png](output_32_2742.png)


    0.9677085056277788
    


![png](output_32_2744.png)


    0.9730631861370538
    


![png](output_32_2746.png)


    0.9778337102686989
    


![png](output_32_2748.png)


    0.9649978388611926
    


![png](output_32_2750.png)


    0.9757596177944703
    


![png](output_32_2752.png)


    0.967778705263457
    


![png](output_32_2754.png)


    0.9669557710972556
    


![png](output_32_2756.png)


    0.9676954439279013
    


![png](output_32_2758.png)


    0.9669825332901485
    


![png](output_32_2760.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 86 	Training Loss: 0.104416 	Validation Loss: 0.360933
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9750936157702752
    


![png](output_32_2765.png)


    0.9374079590327432
    


![png](output_32_2767.png)


    0.9753528291595235
    


![png](output_32_2769.png)


    0.9791383078426663
    


![png](output_32_2771.png)


    0.9673962779197298
    


![png](output_32_2773.png)


    0.9663189616566995
    


![png](output_32_2775.png)


    0.9787458925349657
    


![png](output_32_2777.png)


    0.9632915429151674
    


![png](output_32_2779.png)


    0.9621297214712834
    


![png](output_32_2781.png)


    0.97701840422615
    


![png](output_32_2783.png)


    0.9679066969315959
    


![png](output_32_2785.png)


    0.9729453455733078
    


![png](output_32_2787.png)


    0.981138720431567
    


![png](output_32_2789.png)


    0.9710328089177067
    


![png](output_32_2791.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 87 	Training Loss: 0.096699 	Validation Loss: 0.367404
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9623642954557683
    


![png](output_32_2796.png)


    0.9700892211816952
    


![png](output_32_2798.png)


    0.9427472437210119
    


![png](output_32_2800.png)


    0.9774109760414658
    


![png](output_32_2802.png)


    0.9775838379743997
    


![png](output_32_2804.png)


    0.9780656735899852
    


![png](output_32_2806.png)


    0.9730333704600118
    


![png](output_32_2808.png)


    0.9715903552823002
    


![png](output_32_2810.png)


    0.9663416794316411
    


![png](output_32_2812.png)


    0.9793242475111686
    


![png](output_32_2814.png)


    0.9696877991308913
    


![png](output_32_2816.png)


    0.9653976055470739
    


![png](output_32_2818.png)


    0.9679556158922272
    


![png](output_32_2820.png)


    0.9750994495252525
    


![png](output_32_2822.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 88 	Training Loss: 0.110794 	Validation Loss: 0.340838
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9729151165844381
    


![png](output_32_2827.png)


    0.9664416240437512
    


![png](output_32_2829.png)


    0.9733912589228453
    


![png](output_32_2831.png)


    0.9696135154152556
    


![png](output_32_2833.png)


    0.9739168679859974
    


![png](output_32_2835.png)


    0.972806101810997
    


![png](output_32_2837.png)


    0.970664076380902
    


![png](output_32_2839.png)


    0.9752646625386621
    


![png](output_32_2841.png)


    0.9614057007461636
    


![png](output_32_2843.png)


    0.9670933096820866
    


![png](output_32_2845.png)


    0.9792927999681367
    


![png](output_32_2847.png)


    0.9678046555223876
    


![png](output_32_2849.png)


    0.9786611832646195
    


![png](output_32_2851.png)


    0.9756391371225563
    


![png](output_32_2853.png)


    0.9689770571477685
    


![png](output_32_2855.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 89 	Training Loss: 0.100768 	Validation Loss: 0.356713
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.972722718003341
    


![png](output_32_2860.png)


    0.9756201903100754
    


![png](output_32_2862.png)


    0.9669539378874641
    


![png](output_32_2864.png)


    0.9650803982038566
    


![png](output_32_2866.png)


    0.8978932143133849
    


![png](output_32_2868.png)


    0.9773748529169439
    


![png](output_32_2870.png)


    0.969547883416904
    


![png](output_32_2872.png)


    0.9684500608744855
    


![png](output_32_2874.png)


    0.9557174673878421
    


![png](output_32_2876.png)


    0.9753493266246456
    


![png](output_32_2878.png)


    0.9646721963830878
    


![png](output_32_2880.png)


    0.9601546146220389
    


![png](output_32_2882.png)


    0.9656813831743258
    


![png](output_32_2884.png)


    0.978521943998029
    


![png](output_32_2886.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 90 	Training Loss: 0.133442 	Validation Loss: 0.413735
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.962387431767638
    


![png](output_32_2891.png)


    0.9770665125316345
    


![png](output_32_2893.png)


    0.9690563264854386
    


![png](output_32_2895.png)


    0.9589315655339791
    


![png](output_32_2897.png)


    0.974891812060465
    


![png](output_32_2899.png)


    0.9538587522314466
    


![png](output_32_2901.png)


    0.9765990553891042
    


![png](output_32_2903.png)


    0.9693016739812655
    


![png](output_32_2905.png)


    0.9725184662476989
    


![png](output_32_2907.png)


    0.9678695942519353
    


![png](output_32_2909.png)


    0.9707667440063583
    


![png](output_32_2911.png)


    0.978419698069284
    


![png](output_32_2913.png)


    0.9763503805559703
    


![png](output_32_2915.png)


    0.9772461948817615
    


![png](output_32_2917.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 91 	Training Loss: 0.110424 	Validation Loss: 0.407709
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9736584994241997
    


![png](output_32_2922.png)


    0.9755780910559837
    


![png](output_32_2924.png)


    0.920851204227078
    


![png](output_32_2926.png)


    0.9715862505137578
    


![png](output_32_2928.png)


    0.9736155990729919
    


![png](output_32_2930.png)


    0.9738755450419825
    


![png](output_32_2932.png)


    0.9748402010601267
    


![png](output_32_2934.png)


    0.9701779833721509
    


![png](output_32_2936.png)


    0.9664653074396758
    


![png](output_32_2938.png)


    0.9735138833263404
    


![png](output_32_2940.png)


    0.9729876828316879
    


![png](output_32_2942.png)


    0.978749825487709
    


![png](output_32_2944.png)


    0.9808543885273431
    


![png](output_32_2946.png)


    0.9756450720027259
    


![png](output_32_2948.png)


    0.9733800795327192
    


![png](output_32_2950.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 92 	Training Loss: 0.105963 	Validation Loss: 0.376769
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9735387148119468
    


![png](output_32_2955.png)


    0.9699209351399711
    


![png](output_32_2957.png)


    0.9619128684226568
    


![png](output_32_2959.png)


    0.9710476726245352
    


![png](output_32_2961.png)


    0.980836176371504
    


![png](output_32_2963.png)


    0.9717686300401513
    


![png](output_32_2965.png)


    0.9738604812454819
    


![png](output_32_2967.png)


    0.9636500696012433
    


![png](output_32_2969.png)


    0.9763095323746351
    


![png](output_32_2971.png)


    0.9652667970924842
    


![png](output_32_2973.png)


    0.9690604053767897
    


![png](output_32_2975.png)


    0.9621479122082753
    


![png](output_32_2977.png)


    0.9772532675957659
    


![png](output_32_2979.png)


    0.9718226211391612
    


![png](output_32_2981.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 93 	Training Loss: 0.104197 	Validation Loss: 0.383497
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.950651184793169
    


![png](output_32_2986.png)


    0.9610557092158307
    


![png](output_32_2988.png)


    0.9718883508588789
    


![png](output_32_2990.png)


    0.9821125720951571
    


![png](output_32_2992.png)


    0.9674676360484336
    


![png](output_32_2994.png)


    0.9570029274282722
    


![png](output_32_2996.png)


    0.976485383601924
    


![png](output_32_2998.png)


    0.9811249042448832
    


![png](output_32_3000.png)


    0.9710935834599874
    


![png](output_32_3002.png)


    0.9672053577903181
    


![png](output_32_3004.png)


    0.9702726835086275
    


![png](output_32_3006.png)


    0.9747907091665506
    


![png](output_32_3008.png)


    0.9693349657206989
    


![png](output_32_3010.png)


    0.9671987143207876
    


![png](output_32_3012.png)


    0.9800028202499624
    


![png](output_32_3014.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 94 	Training Loss: 0.102741 	Validation Loss: 0.420982
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9809220614255671
    


![png](output_32_3019.png)


    0.9746131829873113
    


![png](output_32_3021.png)


    0.9770939690182748
    


![png](output_32_3023.png)


    0.9694599973333529
    


![png](output_32_3025.png)


    0.9680033173940863
    


![png](output_32_3027.png)


    0.9671355723795488
    


![png](output_32_3029.png)


    0.9643767772235519
    


![png](output_32_3031.png)


    0.9567905233014691
    


![png](output_32_3033.png)


    0.971124584078209
    


![png](output_32_3035.png)


    0.9740458277153958
    


![png](output_32_3037.png)


    0.9770705946778968
    


![png](output_32_3039.png)


    0.9584741172084816
    


![png](output_32_3041.png)


    0.965680931315656
    


![png](output_32_3043.png)


    0.9768197401975598
    


![png](output_32_3045.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 95 	Training Loss: 0.111020 	Validation Loss: 0.363935
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9652348697178409
    


![png](output_32_3050.png)


    0.9616509984090568
    


![png](output_32_3052.png)


    0.9762311579373025
    


![png](output_32_3054.png)


    0.9696039246799022
    


![png](output_32_3056.png)


    0.9783700134405804
    


![png](output_32_3058.png)


    0.9705336313401882
    


![png](output_32_3060.png)


    0.9762082568777253
    


![png](output_32_3062.png)


    0.975359143559774
    


![png](output_32_3064.png)


    0.9521284276837156
    


![png](output_32_3066.png)


    0.9617923882963622
    


![png](output_32_3068.png)


    0.9633931588141975
    


![png](output_32_3070.png)


    0.9728425144026709
    


![png](output_32_3072.png)


    0.9793716971298612
    


![png](output_32_3074.png)


    0.9724212089511038
    


![png](output_32_3076.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 96 	Training Loss: 0.092941 	Validation Loss: 0.407176
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9599188811080102
    


![png](output_32_3081.png)


    0.9764029031858407
    


![png](output_32_3083.png)


    0.9760315106223207
    


![png](output_32_3085.png)


    0.9776999042630568
    


![png](output_32_3087.png)


    0.9688378228822464
    


![png](output_32_3089.png)


    0.9808287502300685
    


![png](output_32_3091.png)


    0.9590844296587957
    


![png](output_32_3093.png)


    0.9682368957297309
    


![png](output_32_3095.png)


    0.9590875715557126
    


![png](output_32_3097.png)


    0.9770376276757151
    


![png](output_32_3099.png)


    0.9732918614970006
    


![png](output_32_3101.png)


    0.9603068799946088
    


![png](output_32_3103.png)


    0.9660338248840143
    


![png](output_32_3105.png)


    0.9699778118327276
    


![png](output_32_3107.png)


    0.9689461786000764
    


![png](output_32_3109.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 97 	Training Loss: 0.112096 	Validation Loss: 0.400965
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.970021910402747
    


![png](output_32_3114.png)


    0.9758835075399986
    


![png](output_32_3116.png)


    0.9847575261737916
    


![png](output_32_3118.png)


    0.9735379383863694
    


![png](output_32_3120.png)


    0.979286291376475
    


![png](output_32_3122.png)


    0.9768176903650863
    


![png](output_32_3124.png)


    0.9784249654348794
    


![png](output_32_3126.png)


    0.9650037995559042
    


![png](output_32_3128.png)


    0.9710805723403371
    


![png](output_32_3130.png)


    0.9697149923058224
    


![png](output_32_3132.png)


    0.9710431761770146
    


![png](output_32_3134.png)


    0.9775558427290352
    


![png](output_32_3136.png)


    0.9751807515579036
    


![png](output_32_3138.png)


    0.9631175066846676
    


![png](output_32_3140.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 98 	Training Loss: 0.095231 	Validation Loss: 0.374459
    


      0%|          | 0/719 [00:00<?, ?it/s]


    0.9828983842004854
    


![png](output_32_3145.png)


    0.979050766166639
    


![png](output_32_3147.png)


    0.9792102419846711
    


![png](output_32_3149.png)


    0.9735846982499805
    


![png](output_32_3151.png)


    0.9737468013472554
    


![png](output_32_3153.png)


    0.978526059136227
    


![png](output_32_3155.png)


    0.9722418099378005
    


![png](output_32_3157.png)


    0.9744270694805823
    


![png](output_32_3159.png)


    0.9759011296300729
    


![png](output_32_3161.png)


    0.9783070780011518
    


![png](output_32_3163.png)


    0.9751246334180447
    


![png](output_32_3165.png)


    0.9635159485413646
    


![png](output_32_3167.png)


    0.9717158585430382
    


![png](output_32_3169.png)


    0.9775132205059418
    


![png](output_32_3171.png)


    0.94984960023707
    


![png](output_32_3173.png)



      0%|          | 0/160 [00:00<?, ?it/s]


    Epoch: 99 	Training Loss: 0.099061 	Validation Loss: 0.387047
    


```python
#Loss
plt.plot(train_loss)
plt.plot(valid_loss)
```




    [<matplotlib.lines.Line2D at 0x21d43e377f0>]




![png](output_33_1.png)



```python
# model.load_state_dict(torch.load('model_.pt'))
model.load_state_dict(torch.load('model_best_2.pt'))
```




    <All keys matched successfully>




```python
# test_input_files1=sorted(glob.glob("./test/TrainPlusGAN/train/*.npy",recursive=True))
# test_label_files1=sorted(glob.glob("./test/TrainPlusGAN/label/*.npy",recursive=True))
# test_input_files2=sorted(glob.glob("./test/TrainPlusGAN/train/**/*.dcm",recursive=True))
# test_label_files2=sorted(glob.glob("./test/TrainPlusGAN/label/**/*.png",recursive=True))
# test_input_files=test_input_files1+test_input_files2
# test_label_files=test_label_files1+test_label_files2
# test_input_files = np.array(test_input_files)
# test_label_files = np.array(test_label_files)
```


```python
len(test_input_files)
```




    704




```python
class TestMyDataset(torch.utils.data.Dataset):
    def __init__(self, x_dir,y_dir,augmentation = False):
        super().__init__()
        self.augmentation = augmentation
        self.x_img = x_dir
        self.y_img = y_dir
     

    def __len__(self):
        return len(self.x_img)
    

    def __getitem__(self, idx):
        x_img = self.x_img[idx]
        y_img = self.y_img[idx]
        # Read an image with OpenCV  
        x_img = dcm.read_file(x_img)
        x_img=read_dicom(x_img,400,0)
        x_img=np.transpose(x_img,(2,0,1))
        x_img=x_img.astype(np.float32)
        
        y_img = imread(y_img)
        y_img = resize(y_img,(512,512))*255
        color_im = np.zeros([512, 512, 2])
        for i in range(1,3):
            encode_ = to_binary(y_img, i*1.0, i*1.0)
            color_im[:, :, i-1] = encode_
        color_im = np.transpose(color_im,(2,0,1))
        # Data Augmentation
        if self.augmentation:
            img, mask = augment_imgs_and_masks(x_img, color_im, rot_factor, scale_factor, trans_factor, flip)

        return x_img,color_im,y_img
```


```python
test_dataset = TestMyDataset(test_input_files,test_label_files)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1,shuffle=False)
```


```python
len(test_loader)
```




    704




```python
images,labels,a = next(iter(test_loader))
print(images.shape)
print(labels.shape)
plt.figure(figsize=(16,18))
plt.subplot(1,4,1)
plt.imshow(images[0][0])
plt.subplot(1,4,2)
plt.imshow(labels[0][0])
plt.subplot(1,4,3)
plt.imshow(labels[0][1])
plt.subplot(1,4,4)
plt.imshow(a[0])
plt.show()
```

    torch.Size([1, 3, 512, 512])
    torch.Size([1, 2, 512, 512])
    


![png](output_40_1.png)



```python
cnt =0
Iou=0
model.to(device)

with torch.no_grad(): 
        model.eval()
        for data, labels,a in tqdm(test_loader):
                data, labels = data.to(device), labels.to(device)
                # forward pass: compute predicted outputs by passing inputs to the model
                logits = model(data)
                logits = logits.sigmoid()
                logits = mask_binarization(logits.detach().cpu(), 0.5)
                iouu = compute_iou(logits,labels)
                iouu=np.round(iouu,3)*100
                iouu=np.nan_to_num(iouu)
#                 print(iouu)
                Iou+=iouu

                labels=labels[0].detach().cpu().numpy()
                logits=logits[0].detach().cpu().numpy()
                cnt = cnt+1
                
#                 if cnt %200==0:

#                         plt.figure(figsize=(16,18))
#                         plt.subplot(1,5,1)
#                         plt.imshow(labels[0])
#                         plt.subplot(1,5,2)
#                         plt.imshow(labels[1])
#                         plt.subplot(1,5,3)
#                         plt.imshow(logits[0])
#                         plt.subplot(1,5,4)
#                         plt.imshow(logits[1])
#                         plt.subplot(1,5,5)
#                         plt.imshow(a[0])
#                         plt.show()
print("Iou:",Iou//len(test_loader))


```


      0%|          | 0/704 [00:00<?, ?it/s]


    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    

    Iou: 59.0
    

    C:\Users\mmc\AppData\Local\Temp/ipykernel_11372/1148551319.py:16: RuntimeWarning: invalid value encountered in true_divide
      IoU = intersection / union.astype(np.float32)
    


```python
print(iou)
```

    0.94984960023707
    


```python

```

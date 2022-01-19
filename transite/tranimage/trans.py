#from __future__ import print_function
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

from django.http import JsonResponse
import torchvision.transforms as transforms
import torchvision.models as models

import copy



num_progress = 0



def MakePic(styleUrl,contentUrl,modleUrl,Style_weight,Num_steps):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #GPU加速
    def image_loader(image):
        image = image.convert('RGB')
        image = loader(image).unsqueeze(0)
        return image.to(device, torch.float)
    styleImg = Image.open(styleUrl)
    contentImg = Image.open(contentUrl)
    x=contentImg.size[0]
    y=contentImg.size[1]
    
    loader = transforms.Compose([
        transforms.Resize(min(x,y)),  # 统一图片大小
        transforms.ToTensor()])  # 转换成torch张量

    styleImg = styleImg.resize((x, y), Image.ANTIALIAS)
    style_img = image_loader(styleImg)
    content_img = image_loader(contentImg)


    assert style_img.size() == content_img.size(), "风格图与内容图大小必须一致"
    #现在，让我们创建一个方法，通过重新将图片转换成PIL格式来展示。我们将尝试展示内容和风格图片来确保它们被正确的导入。

    unloader = transforms.ToPILImage()  # 将图片转换成PIL格式



    class ContentLoss(nn.Module):
        #内容损失-表示一层内容间距的加权版本
        def __init__(self, target,):
            super(ContentLoss, self).__init__()
            # 我们从用于动态计算梯度的树中“分离”目标内容：
            # 这是一个声明的值，而不是变量。 
            # 否则标准的正向方法将引发错误。
            self.target = target.detach()#切断反向传播。


        def forward(self, input):
            #计算内容损失
            self.loss = F.mse_loss(input, self.target)
            return input






    def gram_matrix(input):
        #将给定矩阵和它的转置矩阵的乘积，内积数值越大，相关关系越大，两个向量越相似。可以用来度量两个图像风格的差异。
        a, b, c, d = input.size()  # a=batch size(=1)
        # 特征映射 b=number
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # 计算gram结果

        # 我们通过除以每个特征映射中的元素数来“标准化”gram矩阵的值.
        return G.div(a * b * c * d)




    class StyleLoss(nn.Module):
        #风格损失模型

        def __init__(self, target_feature):
            super(StyleLoss, self).__init__()
            self.target = gram_matrix(target_feature).detach()

        def forward(self, input):
            G = gram_matrix(input)
            self.loss = F.mse_loss(G, self.target)
            return input



    cnn = models.vgg19(pretrained=True).features.to(device).eval()#使用预训练的19层的 VGG 神经网络。eval()将网络设置成评估模式。





    #使用vgg的参数来标准化图片的每一个通道，并在图片上进行训练。
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


    # 创建一个模块来规范化输入图像
    # 这样我们就可以轻松地将它放入nn.Sequential中
    class Normalization(nn.Module):
        def __init__(self, mean, std):
            super(Normalization, self).__init__()
            # .view the mean and std to make them [C x 1 x 1] so that they can directly work with image Tensor of shape [B x C x H x W].
            # B is batch size 一次训练所选取的样本数。. C is number of channels 通道数. H is height ， W is width.
            self.mean = torch.tensor(mean).view(-1, 1, 1) # -1表示维数自动判断，对已知的进行reshape
            self.std = torch.tensor(std).view(-1, 1, 1)

        def forward(self, img):
            # normalize img
            return (img - self.mean) / self.std





    # 期望的深度层来计算样式/内容损失：
    content_layers_default = ['conv_4']
    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                                   style_img, content_img,
                                   content_layers=content_layers_default,
                                   style_layers=style_layers_default):
        cnn = copy.deepcopy(cnn)

        # 规范化模块
        normalization = Normalization(normalization_mean, normalization_std).to(device)

        # 只是为了拥有可迭代的访问权限或列出内容/系统损失
        content_losses = []
        style_losses = []

        # 假设cnn是一个`nn.Sequential`，
        # 所以我们创建一个新的`nn.Sequential`来放入应该按顺序激活的模块
        model = nn.Sequential(normalization)

        i = 0  # conv 自增循环计数
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):#判断一个对象是否是一个已知的类型
                i += 1
                name = 'conv_{}'.format(i) #卷积函数，是两个变量在某范围内相乘后求和的结果
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)  #线性整流函数（Rectified Linear Unit, ReLU），又称修正线性单元，斜坡函数
                # 对于我们在下面插入的`ContentLoss`和`StyleLoss`，
                # 本地版本不能很好地发挥作用。所以我们在这里替换不合适的
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)  #最大池化函数
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i) #归一化函数BatchNorm
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in content_layers:
                # 加入内容损失:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # 加入风格损失:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        # 现在我们在最后的内容和风格损失之后剪掉了图层
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses







    input_img = content_img.clone()
    # 如果您想使用白噪声而取消注释以下行：
    #input_img = torch.randn(content_img.data.size(), device=device)






    def get_input_optimizer(input_img):
        #创建一个 PyTorch 的 L-BFGS 优化器optim.LBFGS，作为张量去优化。
        # 此行显示输入是需要渐变的参数
        optimizer = optim.LBFGS([input_img.requires_grad_()])
        return optimizer





    def run_style_transfer(cnn, normalization_mean, normalization_std,
                           content_img, style_img, input_img, num_steps=Num_steps,
                           style_weight=Style_weight, content_weight=1):
        """Run the style transfer."""
        print('Building the style transfer model..')
        model, style_losses, content_losses = get_style_model_and_losses(cnn,
            normalization_mean, normalization_std, style_img, content_img)
        optimizer = get_input_optimizer(input_img)

        print('Optimizing..')
        run = [0]
        
        while run[0] <= num_steps:

            def closure():
                # 更正更新的输入图像的值
                input_img.data.clamp_(0, 1)

                optimizer.zero_grad()
                model(input_img)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()
                global num_progress
                
                run[0] += 1
                num_progress = run[0]  * 100 / num_steps;

                if run[0] % 10 == 0:
                    

                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()

                return style_score + content_score

            optimizer.step(closure)

        # 最后的修正......
        input_img.data.clamp_(0, 1)

        return input_img







    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img)

    image = output.cpu().clone()  
    image = image.squeeze(0)      
    image = unloader(image)
    #torch.save(cnn, modleUrl)
    return image







def show_progress(request):
    
    return JsonResponse(num_progress, safe=False)
 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import models.resnext as resnext

class ColorizeClassifier(nn.Module):
    '''This produce the nicest looking images'''
    def __init__(self, feature_cascade=(256, 128, 64, 32, 16), classes=262, training=True):
        super(ColorizeClassifier, self).__init__()
        fc = feature_cascade
        self.upsample = nn.Upsample(scale_factor=2)
        fe = (1024, 512, 256, 64)
        resnet = models.resnet101(pretrained=training)
        resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1).data)
        res_children = list(resnet.children())
        
        n_shortcut = 16
        
        self.ft1 = nn.Sequential(*res_children[0:3]) #64 filters,
        self.ft2 = nn.Sequential(*res_children[3:5]) # 256 filters
        self.ft3 = res_children[5] # 512 filters
        self.ft4 = res_children[6]# 1024 filters
        
        self.bn4 = nn.BatchNorm2d(fe[0])
        self.conv4 = nn.Conv2d(fe[0], fc[0], kernel_size=3, stride=1, padding=1)
        
        sum3 = fc[0] + fe[1]
        self.bn3 = nn.BatchNorm2d(sum3)
        self.conv3 = nn.Conv2d(sum3, fc[1], kernel_size=3, stride=1, padding=1)
        
        sum2 = fc[1] + fe[2] 
        self.bn2 = nn.BatchNorm2d(sum2)
        self.conv2 = nn.Conv2d(sum2, fc[2], kernel_size=3, stride=1, padding=1)
        
        sum1 = fc[2] + fe[3]
        self.bn1 = nn.BatchNorm2d(sum1)
        self.conv1 = nn.Conv2d(sum1, fc[3], kernel_size=3, stride=1, padding=1)
        
        sum0 = fc[3] + 2 * n_shortcut
        self.bn0 = nn.BatchNorm2d(sum0)
        self.conv0 = nn.Conv2d(sum0, fc[4], kernel_size=3, stride=1, padding=1)
        
        
        

        
        
        
    def freeze_ft(self):
        for ft in (self.ft1, self.ft2, self.ft3, self.ft4):
            for param in ft.parameters():
                param.requires_grad = False
                
    def unfreeze_ft(self):
        for ft in (self.ft1, self.ft2, self.ft3, self.ft4):
            for param in ft.parameters():
                param.requires_grad = True
    
    
    def parameters(self, recurse=True):
        for p in super(ColorizeClassifier, self).parameters():
            if p.requires_grad:
                yield p
    
        
    def forward(self, x):
        
        
        
        x1 = self.ft1(x)
        x2 = self.ft2(x1)
        x3 = self.ft3(x2)
        x4 = self.ft4(x3)
        
        x = self.bn4(x4)
        x = self.conv4(x)
        x = F.relu(x)
        
        x = self.upsample(x)
        x = torch.cat((x, x3), dim=1)
        
        x = self.bn3(x)
        x = self.conv3(x)
        x = F.relu(x)
        
        x = self.upsample(x)
        x = torch.cat((x, x2), dim=1)
        
        x = self.bn2(x)
        x = self.conv2(x)
        x = F.relu(x)
        
        x = self.upsample(x)
        x = torch.cat((x, x1), dim=1)
        
        x = self.bn1(x)
        x = self.conv1(x)
        x = F.relu(x)
        
        x = self.upsample(x)
        x = torch.cat((x, s1, s3), dim=1)
                               
        x = self.bn0(x)       
        x = self.conv0(x)
        x = F.relu(x)
        
        x = self.conv_final(x)              
                               
        return x
      
class ColorizeRegressor(ColorizeClassifier):
    '''This produce the lowest error looking images'''            
    def __init__(self, feature_cascade=(256, 128, 64, 32), training=True):
        super(ColorizeRegressor, self).__init__(feature_cascade, 2, training)
    
    def forward(self, x):
        x = super(ColorizeRegressor, self).forward(x)
        x = F.tanh(x)
        return x

    
class ResNextColorizeClassifier(nn.Module):
    '''This produce the nicest looking images'''
    def __init__(self, feature_cascade=(512, 256, 64, 32, 32), classes=262, training=None):
        super(ResNextColorizeClassifier, self).__init__()
        fc = feature_cascade
        self.upsample = nn.Upsample(scale_factor=2)
        fe = (1024, 512, 256, 64)
        n_shortcut = 8
                               
        resnet = resnext.resnext101_64x4d(pretrained=training)
        resnet.features[0].weight = nn.Parameter(resnet.features[0].weight.sum(dim=1).unsqueeze(1).data)
        res_children = list(resnet.features.children())
        
        self.ft1 = nn.Sequential(*res_children[0:3]) #64 filters,
        self.ft2 = nn.Sequential(*res_children[3:5]) # 256 filters
        self.ft3 = res_children[5] # 512 filters
        self.ft4 = res_children[6]# 1024 filters
        
        self.bn4 = nn.BatchNorm2d(fe[0])
        self.conv4 = nn.Conv2d(fe[0], fc[0], kernel_size=3, stride=1, padding=1)
        
        
        sum3 = fc[0] + fe[1]
        self.bn3 = nn.BatchNorm2d(sum3)
        self.conv3 = nn.Conv2d(sum3, fc[1], kernel_size=3, stride=1, padding=1)
        
        sum2 = fc[1] + fe[2] 
        self.bn2 = nn.BatchNorm2d(sum2)
        self.conv2 = nn.Conv2d(sum2, fc[2], kernel_size=3, stride=1, padding=1)
        
        sum1 = fc[2] + fe[3]
        self.bn1 = nn.BatchNorm2d(sum1)
        self.conv1 = nn.Conv2d(sum1, fc[3], kernel_size=3, stride=1, padding=1)
        
        #sum0 = fc[3] + n_shortcut
        self.bn0 = nn.BatchNorm2d(fc[3])
        self.conv0 = nn.Conv2d(fc[3], classes, kernel_size=3, stride=1, padding=1)
                               
        #self.bn_final = nn.BatchNorm2d(fc[4])
        #self.conv_final = nn.Conv2d(fc[4], classes, kernel_size=3, stride=1, padding=1)                       
                               
        #self.short_conv_1 = nn.Conv2d(1, n_shortcut, kernel_size=1, stride=1)
        #self.short_conv_3 = nn.Conv2d(1, n_shortcut, kernel_size=3, stride=1, padding=1)
                               
        
        
    def freeze_ft(self):
        for ft in (self.ft1, self.ft2, self.ft3, self.ft4):
            for param in ft.parameters():
                param.requires_grad = False
                
    def unfreeze_ft(self):
        for ft in (self.ft1, self.ft2, self.ft3, self.ft4):
            for param in ft.parameters():
                param.requires_grad = True
    
    
    def parameters(self, recurse=True):
        for p in super(ResNextColorizeClassifier, self).parameters():
            if p.requires_grad:
                yield p
    
        
    def forward(self, x):
        #s1 = self.short_conv_1(x)
        #s3 = self.short_conv_3(x)
        
        x1 = self.ft1(x)
        x2 = self.ft2(x1)
        x3 = self.ft3(x2)
        x4 = self.ft4(x3)
        
        x = self.bn4(x4)
        x = self.conv4(x)
        x = F.relu(x)
        
        x = self.upsample(x)
        x = torch.cat((x, x3), dim=1)
        
        x = self.bn3(x)
        x = self.conv3(x)
        x = F.relu(x)
        
        x = self.upsample(x)
        x = torch.cat((x, x2), dim=1)
        
        x = self.bn2(x)
        x = self.conv2(x)
        x = F.relu(x)
        
        x = self.upsample(x)
        x = torch.cat((x, x1), dim=1)
        
        x = self.bn1(x)
        x = self.conv1(x)
        x = F.relu(x)
        
        x = self.upsample(x)
        #x = torch.cat((x, s3), dim=1)
                               
        x = self.bn0(x)       
        x = self.conv0(x)
        #x = F.relu(x)
        
        #x = self.conv_final(x)  
        return x
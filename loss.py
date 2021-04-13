import torch
import torch.nn as nn

from utils import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss
        self.bce = nn.BCEWithLogitsLoss
        self.entropy = nn.CrossEntropyLoss
        self.sigmoid = nn.Sigmoid()

        # Constants

        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, predictions, target, anchors):
        obj = target[..., 0] == 1
        noobj = target[..., 0] == 0

        #No object loss
        no_object_loss = self.bce()(
            (predictions[..., 0:1][noobj]), (target[..., 0:1][noobj])
         )

        #object Loss
        anchors = anchors.reshape(1, 3, 1, 1, 2)# 3 x 2 p_w * exp(t_w)
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5])*anchors], dim=-1)
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        object_loss = self.bce()((predictions[..., 0:1][obj]), (ious*target[..., 0:1][obj])) #都是使用iou


        # Box Coords Loss
        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])
        target[..., 3:5] = torch.log(    #target有变化，与上面相反,为的是更好的梯度
            1e-16 + target[..., 3:5] / anchors #加上1e-16是让log接近0时更好
        )
        box_loss = self.mse()((predictions[..., 1:5][obj]), (target[..., 1:5][obj]))

        #class Loss
        class_loss = self.entropy()(
            (predictions[..., 5:][obj]), (target[..., 5][obj].long()), 
        )

        return(
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )
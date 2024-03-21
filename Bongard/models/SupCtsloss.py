import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):

    def __init__(self, temperature=0.07, scale_by_temperature=False):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features, labels=None, mask=None):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        features = F.normalize(features, p=2, dim=2)
        batch_size = features.shape[0]
        bank_size = features.shape[1]  

        if labels is not None and mask is not None: 
            raise ValueError('Cannot define both `labels` and `mask`') 
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(batch_size,-1, 1).to(device)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.permute(0,2,1)).float().to(device)
        else:
            mask = mask.float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.permute(0,2,1)),
            self.temperature)  # 计算两两样本间点乘相似度
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=2, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)

 
        ident = torch.eye(bank_size).unsqueeze(dim=2)
        for i in range(batch_size - 1):
            ident = torch.dstack((ident,torch.eye(bank_size).unsqueeze(dim=2)))
        logits_mask = torch.ones_like(mask).to(device) - ident.permute(2,0,1).to(device)  
        positives_mask = mask * logits_mask
        negatives_mask = 1. - mask

        num_positives_per_row  = torch.sum(positives_mask , axis=2)     
        denominator = torch.sum(
        exp_logits * negatives_mask, axis=2, keepdims=True) + torch.sum(
            exp_logits * positives_mask, axis=2, keepdims=True)  
        
        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")
        

        log_probs = torch.sum(
            log_probs*positives_mask , axis=2)/ num_positives_per_row

        # loss
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        

        loss_m = loss.sum()/torch.nonzero(loss).shape[0]

        return loss_m, loss

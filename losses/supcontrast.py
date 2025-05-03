from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """
    Supervised contrastive loss
    """
    def __init__(self):
        super(SupConLoss, self).__init__()
        self.temperature = 1.0

    def forward(self, text_features, image_features, t_label, i_targets):
        batch_size = text_features.shape[0] 
        batch_size_N = image_features.shape[0] 
        mask = torch.eq(t_label.unsqueeze(1).expand(batch_size, batch_size_N),
                        i_targets.unsqueeze(0).expand(batch_size, batch_size_N)).float().to(t_label.device)

        logits = torch.div(torch.matmul(text_features, image_features.T), self.temperature)
        # For numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach() 
        exp_logits = torch.exp(logits) 
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) 
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - mean_log_prob_pos.mean()

        return loss


# Test
if __name__ == '__main__':
    loss_fun = SupConLoss()
    image_features = torch.randn(3, 5, requires_grad=True)
    text_features = torch.randn(3, 5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(5)  # Label

    loss_i2t = loss_fun(image_features, text_features, target, target)
    loss_t2i = loss_fun(text_features, image_features, target, target)

    loss = loss_i2t + loss_t2i
    print(loss)

    # loss.backward()
    # optimizer.step()

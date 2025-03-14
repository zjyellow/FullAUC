"""
    A re-impletment for prototypical triplet loss(TGRS 2024).
"""

import torch
import torch.nn as nn
class GPTLLoss(nn.CrossEntropyLoss):
    def __init__(self, **options):
        super(GPTLLoss, self).__init__()
        self.lbda_global = 1.0
        self.lbda_local = 1.0
        self.num_classes = options['num_classes']
        self.anchors = torch.diag(torch.Tensor([3. for i in range(self.num_classes)])).cuda() # set T = 3 in raw paper.
        self.margin_global=2.0
        self.margin_local=1.5

    def distance_classifier(self, x, anchors=None):
        # Calculates euclidean distance from x to each class anchor
        # Returns n x m array of distance from input of batch_size n to anchors of size m
        if anchors is None: anchors = self.anchors
        n = x.size(0) # 128
        m = self.num_classes # 6
        d = self.num_classes # 6
        x = x.unsqueeze(1).expand(n, m, d).double() # 128x6 -> 128x1x6 ->128x6x6
        anchors = anchors.unsqueeze(0).expand(n, m, d)
        dists = torch.norm(x-anchors, 2, 2) # 128x6
        return dists

    def forward(self, x, y, labels=None, train_mode=True):
        embedding, logits, labels = x, y, labels # logits: 128x6
        predictions = logits.data.max(1)[1]
        if not train_mode: 
            logits = self.distance_classifier(logits, self.anchors)
            return logits, 0

        global_losses = []
        local_losses = []
        global_losses = torch.tensor([]).cuda()
        local_losses = torch.tensor([]).cuda()

        for cl in range(self.num_classes):
            # global PTL
            TP_mask = (predictions == cl) & (labels == cl)
            FP_mask = (predictions == cl) & (labels != cl)
            TP_logits = logits[TP_mask]
            FP_logits = logits[FP_mask]

            if len(TP_logits) == 0 or len(FP_logits) == 0:
                continue
            anchor = self.anchors[cl]

            TP_distances = torch.norm(TP_logits - anchor, p=2, dim=1) ** 2
            FP_distances = torch.norm(FP_logits - anchor, p=2, dim=1) ** 2


            global_loss = TP_distances.unsqueeze(1) - FP_distances.unsqueeze(0) + self.margin_global
            global_loss = global_loss.flatten()
            global_loss = global_loss[global_loss>0]
            global_losses = torch.cat((global_losses,global_loss),dim=0)


            # local PTL
            FN_mask = (predictions != cl) & (labels == cl)
            FN_logits = logits[FN_mask]

            TN_mask = (predictions != cl) & (labels != cl)
            TN_logits = logits[TN_mask]
            FP_logits = torch.cat((FP_logits,TN_logits),dim=0)

            if len(FN_logits) == 0 or len(TP_logits) == 0 or len(FP_logits) == 0:
                continue

            local_FN_distance = torch.norm(FN_logits.unsqueeze(0) - TP_logits.unsqueeze(1), p=2, dim=2) ** 2
            local_FP_distance = torch.norm(FP_logits.unsqueeze(0) - TP_logits.unsqueeze(1), p=2, dim=2) ** 2

            local_loss = local_FN_distance.unsqueeze(2) - local_FP_distance.unsqueeze(1) + self.margin_local
            local_loss = local_loss.flatten()

            local_loss = local_loss[local_loss>0]
            local_losses = torch.cat((local_losses,local_loss),dim=0)

        if len(global_losses) > 0 and len(local_losses) > 0:
            final_global_loss = torch.mean(global_losses)
            final_local_loss = torch.mean(local_losses)
            final_loss = final_global_loss*self.lbda_global + final_local_loss *self.lbda_local
        elif len(global_losses) > 0 and len(local_losses) == 0:
            final_loss = torch.mean(global_losses)
        else:
            final_loss = torch.tensor(0.0, requires_grad=True)
        
        return logits, final_loss
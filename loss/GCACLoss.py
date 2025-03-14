import torch
import torch.nn as nn

class GCACLoss(nn.CrossEntropyLoss):

    def __init__(self, **options):
        super(GCACLoss, self).__init__()
        self.lbda = 0.1
        self.num_classes = options['num_classes']
        self.anchors = torch.diag(torch.Tensor([10 for i in range(self.num_classes)])).cuda()

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

    def forward(self, x, y, labels=None, anchors=None, train_mode=True):
        '''Returns CAC loss, as well as the Anchor and Tuplet loss components separately for visualisation.'''
        distances = y # logits
        gt = labels

        distances = self.distance_classifier(distances, anchors)
        if gt is None or not train_mode: return distances, 0

        true = torch.gather(distances, 1, gt.view(-1, 1)).view(-1)
        non_gt = torch.Tensor([[i for i in range(self.num_classes) if gt[x] != i] for x in range(len(distances))]).long().cuda() 
        others = torch.gather(distances, 1, non_gt)
        
        anchor_loss = torch.mean(true)

        tuplet_loss = torch.exp(-others+true.unsqueeze(1))
        tuplet_loss = torch.mean(torch.log(1+torch.sum(tuplet_loss, dim = 1)))

        loss = self.lbda*anchor_loss + tuplet_loss

        return distances, loss
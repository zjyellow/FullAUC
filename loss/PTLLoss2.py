"""
    A re-impletment for prototypical triplet loss(TGRS 2024).
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

class PTLLoss2(nn.CrossEntropyLoss):
    def __init__(self, **options):
        super(PTLLoss2, self).__init__()
        self.lbda_global = 1.0
        self.lbda_local = 1.0
        self.num_classes = options['num_classes']

        self.magnitude = 3
        self.anchors = torch.diag(torch.Tensor([self.magnitude for i in range(self.num_classes)])).cuda() # set T = 3 in raw paper.

        self.margin_global=1.5
        self.margin_local=1.5
        distance_function = nn.PairwiseDistance(p=2)
        self.pdist = distance_function.cuda()
        self.triplet_loss_global = nn.TripletMarginWithDistanceLoss(margin=self.margin_global,distance_function=self.pdist)
        self.triplet_loss_local = nn.TripletMarginWithDistanceLoss(margin=self.margin_local,distance_function=self.pdist)

    def distance_classifier(self, x, anchors=None):
        # Calculates euclidean distance from x to each class anchor
        # Returns n x m array of distance from input of batch_size n to anchors of size m
        if anchors is None: anchors = self.anchors
        n = x.size(0)
        m = self.num_classes
        d = self.num_classes
        x = x.unsqueeze(1).expand(n, m, d).double()
        anchors = anchors.unsqueeze(0).expand(n, m, d)
        dists = torch.norm(x-anchors, 2, 2) ** 2
        return dists

    def forward(self, x, y, labels=None, train_mode=True):
        x_embeddings = torch.Tensor(y).cuda() # logits
        y_embeddings = torch.Tensor(labels).cuda() # labels
        
        if not train_mode: # evaluation & test
            return self.distance_classifier(x_embeddings, self.anchors), 0

        x_pred = torch.argmax(x_embeddings, dim=1).cuda()

        ###################################
        # LOCAL ATTRACTION AND REPULSION #
        ###################################

        triplet_loss_local = Variable(torch.Tensor([0.])).cuda()
        total_local = 0

        for c in range(self.num_classes):
            # Select anchors, positives, and negatives
            # Local positives and negatives
            anchor_indices = (x_pred == c) & (y_embeddings == c)   # TP as anchor
            positive_indices = (x_pred != c) & (y_embeddings == c) # FN as postive
            negative_indices = (y_embeddings != c)                 # FP+TN as negative
            
            if positive_indices.sum() == 0 or negative_indices.sum() == 0:
                continue
            
            anchors_local = x_embeddings[anchor_indices]
            positives_local = x_embeddings[positive_indices]
            negatives_local = x_embeddings[negative_indices]

            # Random sample triplets
            max_triplets = min(len(anchors_local), len(positives_local), len(negatives_local))
            anchors_local = anchors_local[torch.randperm(len(anchors_local))[:max_triplets]]
            positives_local = positives_local[torch.randperm(len(positives_local))[:max_triplets]]
            negatives_local = negatives_local[torch.randperm(len(negatives_local))[:max_triplets]]

            triplet_loss = self.triplet_loss_local(anchors_local, positives_local, negatives_local)

            if torch.isnan(triplet_loss).any():
                # total_local += 1
                continue
            else:
                triplet_loss_local += triplet_loss
                total_local += 1

        if total_local > 0:
            triplet_loss_local /= total_local

        ###################################
        # GLOBAL ATTRACTION AND REPULSION #
        ###################################

        triplet_loss_global = Variable(torch.tensor([0.])).cuda()
        total_global = 0
        
        for c in range(self.num_classes):
            anchors_global = self.anchors[c].unsqueeze(0)  # anchor
            
            # Global positives and negatives
            positive_indices = (x_pred == c) & (y_embeddings == c) # TP
            negative_indices = (x_pred == c) & (y_embeddings != c) # FP

            if positive_indices.sum() == 0 or negative_indices.sum() == 0:
                continue
            
            positives_global = x_embeddings[positive_indices]
            negatives_global = x_embeddings[negative_indices]

            # Random sample triplets
            max_triplets = min(len(positives_global), len(negatives_global))
            positives_global = positives_global[torch.randperm(len(positives_global))[:max_triplets]]
            negatives_global = negatives_global[torch.randperm(len(negatives_global))[:max_triplets]]

            triplet_loss = self.triplet_loss_global(anchors_global, positives_global, negatives_global)
            
            if torch.isnan(triplet_loss).any():
                # total_global += 1
                continue
            else:
                triplet_loss_global += triplet_loss
                total_global += 1

        if total_global > 0:
            triplet_loss_global /= total_global
        
        final_loss = triplet_loss_global.to(dtype=torch.float64) * self.lbda_global + triplet_loss_local.to(dtype=torch.float64) * self.lbda_local

        return x_embeddings, final_loss
    
if __name__ == '__main__':
    x = torch.tensor(
        [[0.5951, 0.4342, 0.4731, 0.4656, 0.4407, 0.4021],
        [0.3360, 0.5168, 0.4475, 0.6299, 0.6118, 0.3951],
        [0.5165, 0.4712, 0.4737, 0.5123, 0.5481, 0.3801],
        [0.4928, 0.4465, 0.5425, 0.4833, 0.5285, 0.4195],
        [0.5201, 0.4950, 0.4579, 0.4595, 0.5422, 0.3908],
        [0.5077, 0.5066, 0.4137, 0.4785, 0.5323, 0.3967],
        [0.5051, 0.4419, 0.3157, 0.6200, 0.5835, 0.3448],
        [0.6136, 0.4744, 0.4954, 0.4587, 0.4777, 0.4022],
        [0.6015, 0.4724, 0.4695, 0.4225, 0.4559, 0.4194],
        [0.4309, 0.4900, 0.5568, 0.6452, 0.6014, 0.4617],
        [0.5848, 0.4842, 0.4843, 0.4889, 0.5350, 0.4174],
        [0.4614, 0.4386, 0.5957, 0.5368, 0.6162, 0.4999],
        [0.5681, 0.4705, 0.4984, 0.4960, 0.5034, 0.4368],
        [0.3903, 0.3799, 0.4156, 0.4634, 0.6399, 0.4083],
        [0.4472, 0.4010, 0.3953, 0.5626, 0.6209, 0.4532],
        [0.5440, 0.4712, 0.4801, 0.4884, 0.4884, 0.3563],
        [0.5578, 0.4912, 0.4550, 0.4948, 0.5210, 0.4028],
        [0.4328, 0.5105, 0.4360, 0.5159, 0.5896, 0.3951],
        [0.5555, 0.4729, 0.4788, 0.4680, 0.4685, 0.3742],
        [0.4340, 0.4763, 0.5304, 0.5458, 0.6286, 0.4124],
        [0.5189, 0.4513, 0.4982, 0.4428, 0.5402, 0.3968],
        [0.4963, 0.4287, 0.4723, 0.4159, 0.5596, 0.4294],
        [0.5734, 0.4121, 0.5417, 0.4743, 0.4832, 0.4353],
        [0.5003, 0.4320, 0.4920, 0.4767, 0.5690, 0.4269],
        [0.5479, 0.4034, 0.4802, 0.4672, 0.5003, 0.4405],
        [0.5595, 0.4617, 0.4886, 0.4584, 0.5203, 0.4155],
        [0.5118, 0.5142, 0.4038, 0.5099, 0.6592, 0.4420],
        [0.4019, 0.3786, 0.3440, 0.5073, 0.6138, 0.3637],
        [0.4435, 0.4494, 0.3378, 0.4977, 0.5458, 0.3827],
        [0.5453, 0.4364, 0.4789, 0.4465, 0.4734, 0.3968],
        [0.4852, 0.4568, 0.4183, 0.5158, 0.5920, 0.4590],
        [0.5686, 0.4301, 0.4038, 0.4372, 0.7011, 0.4677],
        [0.4411, 0.4689, 0.5826, 0.5668, 0.6655, 0.5071],
        [0.4289, 0.3858, 0.3525, 0.5594, 0.5505, 0.3827],
        [0.5375, 0.5188, 0.4506, 0.4727, 0.5908, 0.3875],
        [0.5364, 0.4184, 0.5315, 0.5206, 0.5850, 0.4281],
        [0.4116, 0.2736, 0.6786, 0.6911, 0.7217, 0.4514],
        [0.5216, 0.3343, 0.4931, 0.4932, 0.5201, 0.4795],
        [0.4532, 0.3674, 0.5407, 0.4804, 0.5479, 0.3804],
        [0.5424, 0.3845, 0.4666, 0.4401, 0.4546, 0.4498],
        [0.5032, 0.4233, 0.4802, 0.4662, 0.4763, 0.3912],
        [0.4340, 0.3484, 0.3560, 0.6066, 0.6073, 0.3885],
        [0.4713, 0.4179, 0.4832, 0.5169, 0.5222, 0.4398],
        [0.4978, 0.4366, 0.5232, 0.5467, 0.6161, 0.4823],
        [0.5242, 0.4298, 0.4871, 0.4645, 0.5149, 0.4213],
        [0.5124, 0.4188, 0.4732, 0.5417, 0.5632, 0.4264],
        [0.4565, 0.4854, 0.4328, 0.5108, 0.4971, 0.4119],
        [0.4877, 0.4888, 0.4478, 0.4687, 0.5302, 0.4422],
        [0.5133, 0.4211, 0.3956, 0.4520, 0.5208, 0.3980],
        [0.4862, 0.4736, 0.4356, 0.4413, 0.5599, 0.4448],
        [0.4784, 0.4105, 0.5580, 0.5635, 0.6429, 0.4463],
        [0.4681, 0.4643, 0.4398, 0.5108, 0.5267, 0.3970],
        [0.5456, 0.4502, 0.5091, 0.4749, 0.4965, 0.4080],
        [0.4892, 0.5068, 0.4746, 0.5768, 0.5312, 0.4293],
        [0.4933, 0.3878, 0.4940, 0.4906, 0.5482, 0.4214],
        [0.4378, 0.5467, 0.4882, 0.5257, 0.5977, 0.4586],
        [0.4602, 0.4581, 0.4318, 0.5012, 0.5729, 0.4391],
        [0.5134, 0.5056, 0.4869, 0.5120, 0.5662, 0.4477],
        [0.6152, 0.4014, 0.4245, 0.5063, 0.6309, 0.4178],
        [0.4692, 0.4499, 0.4403, 0.4313, 0.5686, 0.4200],
        [0.5093, 0.4361, 0.4614, 0.4551, 0.5513, 0.4360],
        [0.5243, 0.5178, 0.5195, 0.4831, 0.5651, 0.4392],
        [0.5031, 0.4664, 0.4492, 0.5138, 0.4864, 0.3658],
        [0.5043, 0.4681, 0.4819, 0.4833, 0.6503, 0.4185]]
    )
    y = torch.tensor(
        [3, 2, 3, 0, 3, 3, 4, 1, 5, 2, 5, 3, 0, 5, 1, 4, 5, 3, 1, 2, 5, 5, 1, 0,
        1, 3, 0, 2, 3, 1, 4, 3, 3, 4, 0, 2, 4, 5, 5, 1, 0, 1, 5, 3, 0, 3, 5, 4,
        0, 3, 5, 2, 3, 1, 4, 5, 1, 3, 0, 0, 3, 2, 0, 3]
    )
    # logits = x
    # labels = y
    # anchors = torch.diag(torch.Tensor([3 for i in range(6)]))
    # predictions = logits.data.max(1)[1]

    # num_classes = 6
    # global_losses = torch.tensor([])
    # local_losses = torch.tensor([])

    # for cl in range(num_classes):
    #     TP_mask = (predictions == cl) & (labels == cl)
    #     FP_mask = (predictions == cl) & (labels != cl)
    #     TP_logits = logits[TP_mask]
    #     FP_logits = logits[FP_mask]

    #     if len(TP_logits) == 0 or len(FP_logits) == 0:
    #         continue
    #     anchor = anchors[cl]

    #     TP_distances = torch.norm(TP_logits - anchor, p=2, dim=1) ** 2
    #     FP_distances = torch.norm(FP_logits - anchor, p=2, dim=1) ** 2


    #     global_loss = TP_distances.unsqueeze(1) - FP_distances.unsqueeze(0) + 0.0
    #     global_loss = global_loss.flatten()

    #     global_loss = global_loss[global_loss>0]
    #     global_losses = torch.cat((global_losses,global_loss),dim=0)


    #     FN_mask = (predictions != cl) & (labels == cl)
    #     FN_logits = logits[FN_mask]

    #     TN_mask = (predictions != cl) & (labels != cl)
    #     TN_logits = logits[TN_mask]

    #     print(FP_logits.size())
    
    #     print(TN_logits.size())
    #     FP_logits = torch.cat((FP_logits,TN_logits),dim=0)
    #     print(FP_logits.size())

    #     if len(FN_logits) == 0 or len(TP_logits) == 0 or len(FP_logits) == 0:
    #         continue

    #     local_FN_distance = torch.norm(FN_logits.unsqueeze(0) - TP_logits.unsqueeze(1), p=2, dim=2) ** 2
    #     local_FP_distance = torch.norm(FP_logits.unsqueeze(0) - TP_logits.unsqueeze(1), p=2, dim=2) ** 2

    #     local_loss = local_FN_distance.unsqueeze(2) - local_FP_distance.unsqueeze(1) + 0.5
    #     local_loss = local_loss.flatten()

    #     local_loss = local_loss[local_loss>0]
    #     local_losses = torch.cat((local_losses,local_loss),dim=0)


    # if len(global_losses) > 0 and len(local_losses) > 0:
    #     final_global_loss = torch.mean(global_losses)
    #     print(final_global_loss)
    #     final_local_loss = torch.mean(local_losses)
    #     print(final_local_loss)
    #     final_loss = final_global_loss*1.0 + final_local_loss *1.0
    # else:
    #     final_loss = torch.tensor(0.0)



################################################
    x_embeddings = torch.Tensor(x).cuda()
    y_embeddings = torch.Tensor(y).cuda()
    margin_global=1.5
    margin_local=0
    magnitude = 3.0
    num_classes = 6
    anchors = torch.diag(torch.Tensor([magnitude for _ in range(num_classes)])).cuda()
    Ftriplet_loss_global = nn.TripletMarginWithDistanceLoss(margin=margin_global)
    Ftriplet_loss_local = nn.TripletMarginWithDistanceLoss(margin=margin_local)
    x_pred = torch.argmax(x_embeddings, dim=1).cuda()
    
    ###################################
    # LOCAL ATTRACTION AND REPULSION #
    ###################################

    triplet_loss_local = Variable(torch.Tensor([0])).cuda()
    total_local = 0


    for c in range(num_classes):
        # Select anchors, positives, and negatives
        # print(c)
        

        # Local positives and negatives
        anchor_indices = (x_pred == c) & (y_embeddings == c)
        positive_indices = (x_pred != c) & (y_embeddings == c)
        negative_indices = (y_embeddings != c)
        
        if positive_indices.sum() == 0 or negative_indices.sum() == 0:
            continue
        
        anchors_local = x_embeddings[anchor_indices]
        # print(anchors_local)
        positives_local = x_embeddings[positive_indices]
        negatives_local = x_embeddings[negative_indices]

        # Random sample triplets
        max_triplets = min(len(anchors_local), len(positives_local), len(negatives_local))
        anchors_local = anchors_local[torch.randperm(len(anchors_local))[:max_triplets]]
        positives_local = positives_local[torch.randperm(len(positives_local))[:max_triplets]]
        negatives_local = negatives_local[torch.randperm(len(negatives_local))[:max_triplets]]

        triplet_loss = Ftriplet_loss_local(anchors_local, positives_local, negatives_local)

        if torch.isnan(triplet_loss).any():
            # total_local += 1
            continue
        else:
            triplet_loss_local += triplet_loss
            total_local += 1

    if total_local > 0:
        triplet_loss_local /= total_local

    ###################################
    # GLOBAL ATTRACTION AND REPULSION #
    ###################################

    triplet_loss_global = Variable(torch.Tensor([0])).cuda()
    total_global = 0
    
    for c in range(num_classes):
        anchors_global = anchors[c].unsqueeze(0)  # Reshape for batch processing
        
        # Global positives and negatives
        positive_indices = (x_pred == c) & (y_embeddings == c)
        negative_indices = (x_pred == c) & (y_embeddings != c)

        if positive_indices.sum() == 0 or negative_indices.sum() == 0:
            continue
        
        positives_global = x_embeddings[positive_indices]
        negatives_global = x_embeddings[negative_indices]

        # Random sample triplets
        max_triplets = min(len(positives_global), len(negatives_global))
        positives_global = positives_global[torch.randperm(len(positives_global))[:max_triplets]]
        negatives_global = negatives_global[torch.randperm(len(negatives_global))[:max_triplets]]

        triplet_loss = Ftriplet_loss_global(anchors_global, positives_global, negatives_global)
        print(triplet_loss)
        if torch.isnan(triplet_loss).any():
            # total_global += 1
            continue
        else:
            triplet_loss_global += triplet_loss
            total_global += 1

    if total_global > 0:
        triplet_loss_global /= total_global
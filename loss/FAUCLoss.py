import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.Dist import Dist

class FAUCLoss(nn.CrossEntropyLoss):
    def __init__(self, **options):
        super(FAUCLoss, self).__init__()
        self.gama = options['gama']
        print('gama={}'.format(self.gama))

    def forward(self, x, y, labels=None): # x: embeddings, y: logits
        logits = y
        logits = torch.sigmoid(logits)
        if labels is None: return logits, 0
        y = labels
        y_onehot = F.one_hot(y, logits.shape[1]+1)
        y_onehot = y_onehot[:, :-1].bool()
        scores_true = torch.masked_select(logits, y_onehot)
        scores_margin = torch.max(torch.masked_fill(logits, y_onehot, 0), dim=-1)[0]
        margain = scores_true.unsqueeze(-1) - scores_margin.unsqueeze(0) - self.gama
        # margain = scores_true - scores_margin - self.gama
        margain = torch.masked_select(margain, margain < 0)
        loss = -torch.sum(torch.pow(margain, 1)) / scores_true.shape[0] / scores_margin.shape[0]  # hinge loss
        # loss = torch.sum(torch.pow(margain, 2)) / scores_true.shape[0] / scores_margin.shape[0] # square hingeloss
        # loss = torch.sum(torch.pow(margain, 2)) / margain.shape[0]
        return logits, loss

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
    y_onehot = F.one_hot(y, x.shape[1])
    y_onehot = y_onehot[:, :-1].bool()
    scores_true = torch.masked_select(x, y_onehot)
    
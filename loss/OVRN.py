import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.Dist import Dist

class OVRN(nn.CrossEntropyLoss):
    def __init__(self, **options):
        super(OVRN, self).__init__()
        # self.use_gpu = options['use_gpu']
        # self.weight_pl = float(options['weight_pl'])
        # self.temp = options['temp']
        # self.Dist = Dist(num_classes=options['num_classes'], feat_dim=options['feat_dim'])
        # self.points = self.Dist.centers
        # self.radius = nn.Parameter(torch.Tensor(1))
        # self.radius.data.fill_(0)
        # self.margin_loss = nn.MarginRankingLoss(margin=1.0)
        # self.gama = options['gama']
        # print('gama={}'.format(self.gama))

    def forward(self, x, y, labels=None): # x: embeddings, y: logits
        # sigmoid network
        logits = y
        targets = labels

        batch_size, num_classes, _ = logits.shape
        selected_logits = logits[torch.arange(batch_size), labels]
        

        if labels is None:  # test
            logits = torch.sigmoid(logits)
            scores = torch.zeros((batch_size, num_classes), device=logits.device)
            mask_template = torch.eye(num_classes, device=logits.device)
            for class_idx in range(num_classes):
                row_logits = logits[:, class_idx, :]
                mask = mask_template[class_idx]
                class_score = (mask * torch.log(row_logits + 1e-10) + 
                      (1 - mask) * torch.log(1 - row_logits + 1e-10))
                scores[:, class_idx] = class_score.sum(dim=-1)
            return scores, 0

        probs = torch.sigmoid(logits)
        target_matrix = torch.zeros_like(probs)
        batch_indices = torch.arange(batch_size)
        target_matrix[batch_indices, :, targets] = 1
        loss = -(target_matrix * torch.log(probs + 1e-10) + 
             (1 - target_matrix) * torch.log(1 - probs + 1e-10))
        loss = loss.sum(dim=(1, 2)).mean()
        

        return selected_logits, loss
        # num_classes = logits.shape[1]
        # targets_one_hot = F.one_hot(labels, num_classes=num_classes).float()
        # loss = F.binary_cross_entropy_with_logits(logits, targets_one_hot)

        # # train
        # y = labels
        # y_onehot = F.one_hot(y, logits.shape[1])
        # y_onehot = y_onehot.bool()
        # # postive -- log P(tj | xj)
        # scores_true = torch.masked_select(logits, y_onehot).unsqueeze(-1)
        # scores_true = torch.log(scores_true)

        # # negative -- log (1- P(yj!=tj | xj))
        # scores_false = torch.masked_fill(logits, y_onehot, 0)
        # scores_false = torch.ones_like(scores_false) - scores_false
        # scores_false = torch.sum(torch.log(scores_false), dim=1, keepdim=True)

        # # binary entropy --  -[log P(tj | xj) + log (1- P(yj!=tj | xj))], mean by batch size N
        # loss = - torch.mean(scores_true + scores_false)

        # if torch.isnan(loss):
        #     print(logits2)
        #     print(logits)
        #     print(scores_true)
        #     print(scores_false)
        #     exit()
            # print(labels)
            # return logits, torch.tensor(0.0,requires_grad=True)
            # return logits, 0
        # return logits2, loss


if __name__ == '__main__':
    logits = torch.tensor(
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
    y_onehot = F.one_hot(y, logits.shape[1])
    y_onehot = y_onehot.bool()
    scores_true = torch.masked_select(logits, y_onehot)
    scores_true = scores_true.unsqueeze(-1)
    # M = logits.shape[1]
    # score_sum = torch.sum(logits,dim=1)
    # M = torch.ones_like(scores_true) * (M - 1)
    # loss = torch.mean(2 * scores_true + M - score_sum)
    scores_false = torch.masked_fill(logits, y_onehot, 0)

    scores_false = torch.ones_like(scores_false) - scores_false
    scores_false = torch.sum(torch.log(scores_false), dim=1,keepdim=True)
    scores_true = torch.log(scores_true)
    loss = - torch.mean(scores_true + scores_false)

    logits = torch.tensor([[0.1373, 0.1368, 0.2607, 0.1547, 0.1977, 0.9327],
        [0.9251, 0.0584, 0.2541, 0.2667, 0.0946, 0.1669],
        [0.1749, 0.7695, 0.1747, 0.1634, 0.4284, 0.3262],
        [0.1976, 0.8797, 0.1597, 0.3747, 0.3199, 0.2647],
        [0.1843, 0.1714, 0.0679, 0.1508, 0.9122, 0.2556],
        [0.0941, 0.2643, 0.3845, 0.8904, 0.1072, 0.2204],
        [0.2575, 0.2962, 0.4200, 0.7960, 0.2225, 0.2574],
        [0.1869, 0.2143, 0.2450, 0.7562, 0.4499, 0.1217],
        [0.2455, 0.4733, 0.7535, 0.2038, 0.3327, 0.1888],
        [0.2261, 0.3023, 0.3673, 0.1962, 0.1290, 0.7453],
        [0.9231, 0.0421, 0.2963, 0.2897, 0.0787, 0.1601],
        [0.1280, 0.1883, 0.1970, 0.8805, 0.1844, 0.1934],
        [0.8680, 0.1014, 0.2737, 0.3324, 0.2047, 0.1365],
        [0.6988, 0.1772, 0.4403, 0.2173, 0.3543, 0.2473],
        [0.2304, 0.1995, 0.2964, 0.8696, 0.2762, 0.1254],
        [0.1210, 0.1307, 0.3913, 0.8951, 0.0957, 0.2600],
        [0.1445, 0.7280, 0.1088, 0.1472, 0.5147, 0.2985],
        [0.1976, 0.1564, 0.0639, 0.2179, 0.9107, 0.1471],
        [0.1733, 0.1701, 0.3278, 0.8085, 0.2949, 0.1746],
        [0.1487, 0.1411, 0.2215, 0.8705, 0.1970, 0.3274],
        [0.9063, 0.0587, 0.2778, 0.2024, 0.1667, 0.2171],
        [0.8307, 0.0723, 0.1886, 0.2463, 0.3635, 0.1296],
        [0.2000, 0.2386, 0.4075, 0.1907, 0.7144, 0.2770],
        [0.2011, 0.3082, 0.3423, 0.1934, 0.1511, 0.7605],
        [0.1942, 0.1956, 0.3727, 0.8800, 0.2028, 0.1890],
        [0.2754, 0.2858, 0.4143, 0.2888, 0.1935, 0.7268],
        [0.8173, 0.1108, 0.1830, 0.4780, 0.2377, 0.2012],
        [0.2726, 0.1286, 0.0214, 0.2216, 0.8533, 0.2980],
        [0.2150, 0.9264, 0.1342, 0.1677, 0.3926, 0.2408],
        [0.1142, 0.1671, 0.1772, 0.8726, 0.1587, 0.2991],
        [0.1430, 0.0965, 0.9341, 0.2254, 0.1556, 0.1668],
        [0.8366, 0.0541, 0.2063, 0.3551, 0.3074, 0.1899],
        [0.2467, 0.9345, 0.1967, 0.1696, 0.2596, 0.3544],
        [0.1822, 0.3011, 0.2355, 0.1985, 0.1657, 0.8839],
        [0.0855, 0.1989, 0.9377, 0.1002, 0.2080, 0.1406],
        [0.2097, 0.2299, 0.1726, 0.7529, 0.2268, 0.3151],
        [0.1460, 0.2644, 0.1194, 0.1345, 0.3417, 0.8718],
        [0.0905, 0.2249, 0.2865, 0.1894, 0.3524, 0.6786],
        [0.1150, 0.1185, 0.3656, 0.0872, 0.0953, 0.9625],
        [0.1998, 0.1537, 0.9041, 0.1376, 0.1949, 0.2495],
        [0.1964, 0.9026, 0.2227, 0.3535, 0.3132, 0.2890],
        [0.1629, 0.1341, 0.1653, 0.8469, 0.3077, 0.1763],
        [0.1407, 0.0586, 0.0502, 0.2502, 0.8998, 0.2491],
        [0.1769, 0.2625, 0.4022, 0.7442, 0.2683, 0.1667],
        [0.1772, 0.3704, 0.3501, 0.3168, 0.2088, 0.7927],
        [0.1195, 0.2342, 0.9249, 0.1005, 0.2727, 0.2037],
        [0.2682, 0.2225, 0.3784, 0.1441, 0.1924, 0.8140],
        [0.9279, 0.0437, 0.2192, 0.2635, 0.1088, 0.2024],
        [0.8616, 0.0865, 0.3155, 0.2496, 0.2219, 0.1379],
        [0.0951, 0.9388, 0.0849, 0.2499, 0.3882, 0.2862],
        [0.2141, 0.1088, 0.2637, 0.1837, 0.3522, 0.7051],
        [0.1134, 0.1769, 0.4253, 0.6340, 0.4197, 0.1171],
        [0.1743, 0.9312, 0.2050, 0.2715, 0.1977, 0.3137],
        [0.2044, 0.1979, 0.3314, 0.8504, 0.2765, 0.2414],
        [0.1846, 0.2066, 0.3027, 0.2515, 0.2326, 0.8483],
        [0.1474, 0.2204, 0.3330, 0.2068, 0.1893, 0.8851],
        [0.8400, 0.0997, 0.2968, 0.2584, 0.2568, 0.2092],
        [0.1076, 0.1851, 0.9218, 0.2313, 0.2033, 0.1481],
        [0.2050, 0.0944, 0.3298, 0.2088, 0.1616, 0.9271],
        [0.1307, 0.1477, 0.0907, 0.2082, 0.8735, 0.2919],
        [0.1816, 0.2025, 0.8793, 0.2944, 0.1513, 0.1407],
        [0.3012, 0.2729, 0.7984, 0.2860, 0.2638, 0.2627],
        [0.8980, 0.1190, 0.2714, 0.3279, 0.1309, 0.1818],
        [0.2114, 0.1483, 0.4011, 0.1908, 0.1825, 0.9177],
        [0.2225, 0.2025, 0.4575, 0.8206, 0.1513, 0.2362],
        [0.0532, 0.1990, 0.2687, 0.1649, 0.3061, 0.8614],
        [0.1573, 0.1845, 0.2245, 0.2073, 0.2370, 0.8133],
        [0.2762, 0.1535, 0.1562, 0.1514, 0.8633, 0.2136],
        [0.2063, 0.1736, 0.2421, 0.1718, 0.2919, 0.7181],
        [0.2030, 0.2448, 0.4182, 0.2167, 0.2677, 0.6457],
        [0.0801, 0.1494, 0.0328, 0.1155, 0.9395, 0.2245],
        [0.2192, 0.2082, 0.0827, 0.2157, 0.8636, 0.2378],
        [0.9297, 0.0326, 0.2414, 0.2464, 0.1002, 0.2657],
        [0.2654, 0.8704, 0.1667, 0.2368, 0.4337, 0.2911],
        [0.2016, 0.8959, 0.1906, 0.2766, 0.2944, 0.3664],
        [0.1119, 0.9251, 0.1100, 0.1777, 0.3995, 0.2155],
        [0.2262, 0.8788, 0.2055, 0.2921, 0.3931, 0.3468],
        [0.1092, 0.1651, 0.9217, 0.2289, 0.0925, 0.2538],
        [0.1323, 0.1124, 0.3019, 0.8189, 0.2628, 0.2873],
        [0.1502, 0.2225, 0.9379, 0.0556, 0.2650, 0.1712],
        [0.8852, 0.1129, 0.2141, 0.3740, 0.1782, 0.1556],
        [0.2073, 0.2350, 0.2247, 0.2052, 0.2466, 0.8677],
        [0.2445, 0.2337, 0.2086, 0.8525, 0.2559, 0.1303],
        [0.1738, 0.2536, 0.3450, 0.1239, 0.3364, 0.7489],
        [0.2982, 0.7541, 0.1524, 0.3561, 0.3255, 0.3502],
        [0.8644, 0.0744, 0.2939, 0.3055, 0.0821, 0.1896],
        [0.1844, 0.3180, 0.2390, 0.2258, 0.3549, 0.7228],
        [0.1487, 0.1444, 0.9310, 0.0828, 0.2289, 0.1166],
        [0.2933, 0.8214, 0.1687, 0.2827, 0.4393, 0.2729],
        [0.2229, 0.8052, 0.1708, 0.3075, 0.3582, 0.2563],
        [0.2335, 0.2054, 0.2707, 0.8317, 0.2167, 0.1846],
        [0.1050, 0.2442, 0.8564, 0.2641, 0.1955, 0.2296],
        [0.2126, 0.2667, 0.7077, 0.4257, 0.2130, 0.1896],
        [0.1627, 0.9118, 0.2143, 0.1511, 0.3100, 0.3292],
        [0.9211, 0.0719, 0.1440, 0.2814, 0.1547, 0.1899],
        [0.1844, 0.1778, 0.2526, 0.1821, 0.2502, 0.8313],
        [0.1409, 0.1705, 0.9150, 0.3228, 0.0998, 0.2146],
        [0.2477, 0.2451, 0.3792, 0.3077, 0.1767, 0.8081],
        [0.2303, 0.2063, 0.8743, 0.2702, 0.2677, 0.1932],
        [0.1709, 0.9395, 0.1178, 0.2653, 0.2600, 0.2786],
        [0.1903, 0.3772, 0.2459, 0.2920, 0.1684, 0.7359],
        [0.2094, 0.1222, 0.8197, 0.1725, 0.4768, 0.1098],
        [0.2159, 0.2205, 0.3991, 0.2448, 0.2385, 0.7290],
        [0.1252, 0.2824, 0.8641, 0.1476, 0.2414, 0.2950],
        [0.1123, 0.8365, 0.0942, 0.2586, 0.5296, 0.2473],
        [0.1880, 0.1840, 0.1725, 0.1168, 0.1425, 0.9110],
        [0.1351, 0.8855, 0.1318, 0.3671, 0.3388, 0.2697],
        [0.3273, 0.9173, 0.1583, 0.1485, 0.2314, 0.2919],
        [0.2288, 0.2152, 0.3025, 0.8347, 0.2623, 0.2606],
        [0.1971, 0.9282, 0.1561, 0.3850, 0.1982, 0.2362],
        [0.8231, 0.0589, 0.1136, 0.1993, 0.3069, 0.2158],
        [0.8246, 0.0933, 0.2157, 0.2937, 0.3620, 0.2236],
        [0.1653, 0.4387, 0.3421, 0.2251, 0.2600, 0.6416],
        [0.2083, 0.1589, 0.1527, 0.8319, 0.2603, 0.1975],
        [0.1697, 0.2284, 0.1745, 0.3061, 0.4165, 0.7832],
        [0.2117, 0.1987, 0.2056, 0.3338, 0.7697, 0.3083],
        [0.0749, 0.2697, 0.2570, 0.1425, 0.2962, 0.7941],
        [0.8963, 0.1092, 0.1753, 0.3129, 0.2293, 0.1469],
        [0.2620, 0.2441, 0.4875, 0.7128, 0.3550, 0.2379],
        [0.1789, 0.8625, 0.2326, 0.4523, 0.1714, 0.3576],
        [0.8580, 0.0817, 0.3563, 0.4051, 0.0833, 0.1988],
        [0.1040, 0.2354, 0.3743, 0.2355, 0.1304, 0.8880],
        [0.2067, 0.2274, 0.2738, 0.3408, 0.2870, 0.7344],
        [0.2662, 0.1724, 0.0793, 0.1839, 0.8661, 0.1846],
        [0.1852, 0.1788, 0.2701, 0.7607, 0.3202, 0.1747],
        [0.2014, 0.0668, 0.1360, 0.2370, 0.8305, 0.3818],
        [0.1427, 0.2405, 0.2727, 0.8150, 0.2774, 0.1822],
        [0.0629, 0.8993, 0.1197, 0.1405, 0.1903, 0.5861]])
    labels = torch.tensor([5, 0, 1, 1, 4, 3, 3, 3, 2, 5, 0, 3, 0, 0, 3, 3, 1, 4, 3, 3, 0, 0, 4, 5,
        3, 5, 0, 4, 1, 3, 2, 0, 1, 5, 2, 3, 5, 5, 5, 2, 1, 3, 4, 3, 5, 2, 5, 0,
        0, 1, 5, 3, 1, 3, 5, 5, 0, 2, 5, 4, 2, 2, 0, 5, 3, 5, 5, 4, 5, 5, 4, 4,
        0, 1, 1, 1, 1, 2, 3, 2, 0, 5, 3, 5, 1, 0, 5, 2, 1, 1, 3, 2, 2, 1, 0, 5,
        2, 5, 2, 1, 5, 2, 5, 2, 1, 5, 1, 1, 3, 1, 0, 0, 5, 3, 5, 4, 5, 0, 3, 1,
        0, 5, 5, 4, 3, 4, 3, 1])
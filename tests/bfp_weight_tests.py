import unittest
from lib import BFPWeight
import torch

class Test_Weight(unittest.TestCase):
    def test_filter_split(self):
        tensor_shape = (8, 3, 4, 4)
        channel_group = 4
        weight = 1.1111*torch.ones((tensor_shape))
        weight[0, 0, 0, 0] = 2**7
        weight[6, 0, 0, 1] = 2**5
        #act = torch.arange(1 * 16 * 32 * 32)
        #act = torch.reshape(act, (1, 16, 32, 32))
        #act = act.float()
        bfp_weight = BFPWeight.transform_weight(weight, 8,8, 1)
        #print(bfp_weight)
        golden = torch.zeros(tensor_shape)
        golden[0, 0, 0, 0] = 2**6-1
        result = torch.equal(bfp_weight, golden)
        self.assertEqual(result, True)

if __name__ == '__main__':
    unittest.main()

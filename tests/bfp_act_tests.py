import unittest
from lib import BFPActivation
import torch

class Test_Act(unittest.TestCase):
    def test_channel_split(self):
        tensor_shape = (1, 8, 4, 4)
        channel_group = 4
        act = 1.1111*torch.ones((tensor_shape))
        act[0, 0, 0, 0] = 2**7
        act[0, 6, 0, 1] = 2**5
        #act = torch.arange(1 * 16 * 32 * 32)
        #act = torch.reshape(act, (1, 16, 32, 32))
        #act = act.float()
        bfp_act = BFPActivation.transform_activation(act, 8,8, 4)
        #print(bfp_act)
        golden = torch.zeros(tensor_shape)
        golden[0, 0, 0, 0] = 2**6 -1
        result = torch.equal(bfp_act, golden)
        self.assertEqual(result, True)

if __name__ == '__main__':
    unittest.main()

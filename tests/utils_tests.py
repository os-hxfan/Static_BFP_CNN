import unittest
from lib import Utils
import torch
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class Test_Utils(unittest.TestCase):

    def test_smooth_hist(self):
        test_tensor = torch.Tensor([[1, 2, 3, 4]])
        test_hist = torch.histc(test_tensor, bins=5, min=0, max=4)
        #print("test_hist:", test_hist)

        test_hist = Utils.smooth_hist(test_hist, eps=0.5)
        golden_hist = torch.Tensor([0.5, 0.875, 0.875, 0.875, 0.875])
        is_equal = torch.equal(test_hist, golden_hist)
        self.assertEqual(is_equal, True)

    def test_find_exp_KL(self):
        test_tensor = torch.Tensor([0, 1, 1, 3, 3, 3, 5, 5])
        #test_tensor = torch.arange(25)
        #test_tensor = torch.reshape(test_tensor, (1, 5, 5))
        test_tensor = test_tensor.float()
        test_exp = Utils.find_exp_KL(test_tensor, 8, 8, num_bins=6)
        golden_exp = torch.Tensor([5])
        print(test_exp)
        is_equal = torch.equal(test_exp, golden_exp)
        self.assertEqual(is_equal, True)
    
        
if __name__ == '__main__':
    unittest.main()

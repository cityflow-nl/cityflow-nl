
import unittest

from comm.cityflow_nl_dataset import CityFlowNLDataset
from tracking.config import get_default_detection_config


class TestCityFlowNLDatasetMethods(unittest.TestCase):

    def test_initialize_dataset(self):
        config = get_default_detection_config()
        dataset = CityFlowNLDataset(config.DATA, num_tracks=10)

    def test_get_item_for_d2(self):
        config = get_default_detection_config()
        config.DATA.FORD2 = True
        dataset = CityFlowNLDataset(config.DATA, num_tracks=10)
        dp = dataset.__getitem__(10)
        self.assertEqual(dp["box"], [784, 486, 131, 95])
        self.assertTrue("annotations" in dp)


if __name__ == '__main__':
    unittest.main()

"""Test for CityFlow NL Dataset for Retrieval MK-II"""
import unittest

from torch.utils.data import DataLoader

from comm.cityflow_nl_dataset_mk2 import (CityFlowNLDataset, CityFlowNLInferenceDataset,
                                          get_default_data_config)


class TestCityFlowNLDatasetMethods(unittest.TestCase):

    def test_initialize_dataset(self):
        config = get_default_data_config()
        CityFlowNLDataset(config, num_tracks=10)

    def test_get_item(self):
        config = get_default_data_config()
        dataset = CityFlowNLDataset(config, num_tracks=20)
        track = dataset.__getitem__(10)
        self.assertEqual(list(track["frame"].shape), [16, 3, 128, 128])
        self.assertEqual(list(track["full_frame"].shape), [16, 3, 1080, 1920])
        self.assertEqual(list(track["label"].shape), [16])

    def test_dataloader(self):
        config = get_default_data_config()
        dataset = CityFlowNLDataset(config, num_tracks=20)
        dataloader = DataLoader(dataset, batch_size=1, num_workers=0)
        for track in dataloader:
            self.assertEqual(list(track["frame"].shape), [1, len(track["flow"][0]), 3, 128, 128])
            self.assertEqual(list(track["label"].shape), [1, len(track["flow"][0])])

    def test_get_optical_flow(self):
        config = get_default_data_config()
        config.USE_OPTICAL_FLOW = True
        dataset = CityFlowNLDataset(config, num_tracks=20)
        dataloader = DataLoader(dataset, batch_size=1, num_workers=0)
        for track in dataloader:
            self.assertEqual(list(track["flow"].shape), [1, len(track["flow"][0]), 3, 128, 128])

    def test_get_inference_data(self):
        test_config = get_default_data_config()
        test_config.SPLIT = "test"
        test_config.JSON_PATH = "data/test-tracks.json"
        test_config.POSITIVE_THRESHOLD = 0.
        dataset = CityFlowNLInferenceDataset(test_config, num_tracks=20)
        dataloader = DataLoader(dataset, batch_size=1, num_workers=0)
        for track in dataloader:
            self.assertEqual(list(track["flow"].shape), [1, len(track["flow"][0]), 3, 128, 128])
            self.assertEqual(list(track["frame"].shape), [1, len(track["flow"][0]), 3, 128, 128])


if __name__ == '__main__':
    unittest.main()

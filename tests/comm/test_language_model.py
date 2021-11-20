"""Language Models Tests."""
import unittest

from comm.language_models import CLIPEmbeddingModel
from retrieval.baseline_mk2.config import get_default_config


class TestLanguageModels(unittest.TestCase):
    """Language Models Tests."""

    def test_clip_model_output_shapes(self):
        """Test if the output shape of CLIP models are correct."""
        model_cfg = get_default_config()
        model = CLIPEmbeddingModel(model_cfg=model_cfg.MODEL).cuda()
        results = model.compute_embedding(
            ["hello, world", "hello fred!", "hello, world", "hello fred! fdsaj jfweiaofjksdaf"])
        self.assertEqual(list(results.shape), [4, 512])


if __name__ == '__main__':
    unittest.main()

"""Language models used in CityFlow-NL baselines."""
import abc

import torch
from transformers import (BertModel, BertTokenizer, CLIPModel, CLIPProcessor,
                          CLIPTokenizer)


class LanguageModel(torch.nn.Module, abc.ABC):
    """Base Language Embedding Model. Abstract Class."""

    def __init__(self, model_cfg) -> None:
        super().__init__()
        self.model_cfg = model_cfg

    @abc.abstractmethod
    def compute_embedding(self, batch_nl):
        """Computes an embedding of the input texts.

        Args:
            batch_nl (torch.Tensor): an input batch of strings.

        Raises:
            NotImplementedError: Need to implemented this method in inherited classes.

        Returns:
            (torch.Tensor): A batch of embeddings.
        """
        raise NotImplementedError("Compute Embedding is not Implemented.")

    def forward(self, batch_nl):
        """Forward pass of the language model. Wraps the compute_embedding
        method. Disabled finetuning of the language models as CityFlow-NL's text
        corpus is relatively smaller than most NLP datasets.

        Args: batch_nl (torch.Tensor): an input batch of strings.

        Returns: (torch.Tensor): A batch of embeddings.
        """
        with torch.no_grad():
            embedding = self.compute_embedding(batch_nl)
        return embedding


class BERTEmbeddingModel(LanguageModel):
    """BERT based language embedding model."""

    def __init__(self, model_cfg) -> None:
        super().__init__(model_cfg)
        self.bert_tokenizer = BertTokenizer.from_pretrained(
            self.model_cfg.BERT_CHECKPOINT)
        self.bert_model = BertModel.from_pretrained(
            self.model_cfg.BERT_CHECKPOINT)

    def compute_embedding(self, batch_nl):
        with torch.no_grad():
            tokens = self.bert_tokenizer.batch_encode_plus(batch_nl, padding='longest',
                                                           return_tensors='pt')

            outputs = self.bert_model(tokens['input_ids'].cuda(),
                                      attention_mask=tokens['attention_mask'].cuda())
            lang_embeds = torch.mean(outputs.last_hidden_state, dim=1)
            return lang_embeds


class CLIPEmbeddingModel(LanguageModel):
    """CLIP based language embedding model."""

    def __init__(self, model_cfg) -> None:
        super().__init__(model_cfg)
        self.clip_model = CLIPModel.from_pretrained(
            self.model_cfg.CLIP_CHECKPOINT)
        self.clip_processor = CLIPProcessor.from_pretrained(
            self.model_cfg.CLIP_CHECKPOINT)
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(
            self.model_cfg.CLIP_CHECKPOINT)
        self.nl_embed_dim = 512
        self.vision_embed_dim = 512

    def compute_embedding(self, batch_nl):
        tokens = self.clip_tokenizer.batch_encode_plus(batch_nl, padding='longest',
                                                       return_tensors='pt')

        outputs = self.clip_model.get_text_features(tokens['input_ids'].cuda(),
                                                    attention_mask=tokens['attention_mask'].cuda())
        return outputs

    def compute_visual_embedding(self, batch_imgs):
        inputs = self.clip_processor(images=batch_imgs, return_tensors="pt")
        outputs = self.clip_model.get_image_features(pixel_values=inputs["pixel_values"].cuda())
        return outputs

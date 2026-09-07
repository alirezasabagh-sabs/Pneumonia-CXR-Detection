import torch

from src.models.patch_mil import AttentionPooling, PatchMILClassifier


def test_attention_pooling_respects_validity_and_prior():
    pool = AttentionPooling(4, attn_hidden=8, mask_strength=1.0, hard_filter_threshold=0.3)
    tokens = torch.randn(2, 5, 4)
    validity = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.float32)
    prior = torch.tensor([[0.9, 0.8, 0.7, 0.0, 0.0], [0.9, 0.1, 0.2, 0.8, 0.7]])
    pooled, weights = pool(tokens, validity, prior, return_weights=True)
    assert pooled.shape == (2, 4)
    assert torch.allclose(weights[0, 3:], torch.zeros(2), atol=1e-6)
    assert torch.all(weights >= 0)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-6)


def test_patch_mil_token_path_shapes():
    model = PatchMILClassifier(
        hidden_size=1024, attn_hidden=128, head_dims=[256, 128, 1],
        context_layers=2, context_heads=8, encoder_type="torchxrayvision",
        mask_strength=2.0, hard_filter_threshold=0.3, skip_encoder_build=True,
    )
    tokens = torch.randn(2, 10, 1024)
    validity = torch.ones(2, 10)
    prior = torch.ones(2, 10)
    logits, attn, critical, instance = model.forward_with_attention_and_instance_from_tokens(
        tokens, validity, mask_prior=prior
    )
    assert logits.shape == (2,)
    assert attn.shape == (2, 10)
    assert critical.shape == (2,)
    assert instance.shape == (2, 10)

import torch
import torch.nn.functional as F

from rosa.attention import mean_pairwise_camera_cosine


def test_linear_pairwise_camera_consistency_matches_explicit_pairs():
    torch.manual_seed(5)
    values = F.normalize(torch.randn(2, 3, 4, 5, 7), dim=-1)
    valid = torch.rand(2, 3, 4, 5) > 0.35
    efficient = mean_pairwise_camera_cosine(values, valid)

    explicit = torch.zeros_like(efficient)
    counts = torch.zeros_like(efficient)
    for left in range(values.shape[2]):
        for right in range(left + 1, values.shape[2]):
            pair_valid = valid[:, :, left] & valid[:, :, right]
            similarity = (
                values[:, :, left] * values[:, :, right]
            ).sum(dim=-1)
            explicit += similarity * pair_valid
            counts += pair_valid
    explicit = torch.where(
        counts > 0,
        explicit / counts.clamp_min(1),
        torch.zeros_like(explicit),
    )

    torch.testing.assert_close(efficient, explicit, atol=2e-6, rtol=2e-6)

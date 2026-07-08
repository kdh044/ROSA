"""OcamCalib (Scaramuzza) forward projection — 3D world point → fisheye pixel.

Matches the cam-frame convention used by RoboSense OV cameras (verified in
sphere_stream/sphere_stream/models/warpers/scaramuzza_equirect_warper.py and
test_ocam.py):
    X = image-row direction (DOWN)
    Y = image-col direction (RIGHT)
    Z = back/up;  forward = -Z

Forward projection (OcamCalib):
    norm  = sqrt(X² + Y²)
    theta = atan2(Z, norm)            # forward ray (Z=-1) → -π/2
    rho   = Σ pol[i] · θ^i            # Horner, ascending order
    x_img = X / norm * rho            # row component
    y_img = Y / norm * rho            # col component
    [u; v; 1] = A_swap · [x_img; y_img; 1]
where A_swap swaps A's first two rows in columns [0:2] (OV-mode).

Output: normalized image coords in [0, 1]. Out-of-FOV / behind-camera /
out-of-image points are set to 2.0 so the deformable aggregation kernel
returns 0 for them.
"""
from __future__ import annotations
import math

import torch

__all__ = ["ocam_project_points", "kb_project_points"]


def _horner(pol: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Evaluate polynomial Σ pol[i]·θ^i (ascending).

    pol:   (..., P) coefficients
    theta: (...)    same leading dims as pol minus the P axis
    """
    P = pol.shape[-1]
    # Bring P-axis to the front for easy iteration.
    coefs = pol.unbind(dim=-1)
    rho = coefs[-1].expand_as(theta).clone()
    for i in range(P - 2, -1, -1):
        rho = rho * theta + coefs[i].expand_as(theta)
    return rho


def ocam_project_points(
    key_points: torch.Tensor,
    metas: dict,
    image_wh: torch.Tensor,
) -> torch.Tensor:
    """Project 3D points (lidar frame) to normalized fisheye pixel coords.

    Args:
        key_points: (B, N_anc, num_pts, 3) or
                    (B, N_cam, N_anc, num_pts, 3) — lidar-frame 3D points.
        metas:      dict with keys
                      "lidar2cam":      (B, N_cam, 4, 4)
                      "ocam_intrinsic": (B, N_cam, 3, 3) — affine A (OV)
                      "ocam_pol":       (B, N_cam, P)    — forward polynomial
                      optional "ocam_lens_fov_deg": float, default 205.0
        image_wh:   (B, N_cam, 2) — (W, H) per cam.

    Returns:
        points_2d: (B, N_cam, N_anc, num_pts, 2) normalized to [0, 1].
                   Invalid points set to 2.0.
    """
    lidar2cam = metas["lidar2cam"]               # (B, N_cam, 4, 4)
    A = metas["ocam_intrinsic"]                  # (B, N_cam, 3, 3)
    pol = metas["ocam_pol"]                      # (B, N_cam, P)
    lens_fov_raw = metas.get("ocam_lens_fov_deg", 205.0)
    if isinstance(lens_fov_raw, torch.Tensor):
        # collated scalar tensor (shape (B,)) — all samples share the same FOV
        lens_fov_deg = float(lens_fov_raw.flatten()[0].item())
    else:
        lens_fov_deg = float(lens_fov_raw)
    theta_max = math.radians(lens_fov_deg / 2.0)

    B = key_points.shape[0]
    N_cam = lidar2cam.shape[1]
    if key_points.dim() == 5:
        if key_points.shape[1] != N_cam:
            raise ValueError(
                "camera-conditioned key_points camera dimension must match "
                f"lidar2cam: {key_points.shape[1]} vs {N_cam}"
            )
        N_anc, num_pts = key_points.shape[2:4]
        ones = key_points.new_ones(B, N_cam, N_anc, num_pts, 1)
        pts_h = torch.cat([key_points, ones], dim=-1)[..., None]
        P_cam = torch.matmul(lidar2cam[:, :, None, None], pts_h).squeeze(-1)
    else:
        N_anc, num_pts = key_points.shape[1:3]
        ones = key_points.new_ones(B, N_anc, num_pts, 1)
        pts_h = torch.cat([key_points, ones], dim=-1)
        pts_h = pts_h[:, None, ..., None]
        P_cam = torch.matmul(lidar2cam[:, :, None, None], pts_h).squeeze(-1)
    X = P_cam[..., 0]                            # row direction (down)
    Y = P_cam[..., 1]                            # col direction (right)
    Z = P_cam[..., 2]                            # forward = -Z

    norm = torch.sqrt(X * X + Y * Y + 1e-12)
    theta_ocam = torch.atan2(Z, norm)            # forward → ~-π/2

    # Compute rho for each (cam, anchor, pt). Broadcast pol (B, N_cam, P) into
    # (B, N_cam, N_anc, num_pts).
    pol_exp = pol[:, :, None, None, :].expand(B, N_cam, N_anc, num_pts, pol.shape[-1])
    rho = _horner(pol_exp, theta_ocam)

    scale = torch.where(
        norm > 1e-8,
        rho / norm.clamp(min=1e-8),
        torch.zeros_like(rho),
    )
    x_img = X * scale                            # row coord on image
    y_img = Y * scale                            # col coord on image

    # OV-mode A_swap application — vectorized over (B, N_cam) then broadcast.
    a11 = A[..., 1, 0][:, :, None, None]
    a12 = A[..., 1, 1][:, :, None, None]
    a13 = A[..., 0, 2][:, :, None, None]
    a21 = A[..., 0, 0][:, :, None, None]
    a22 = A[..., 0, 1][:, :, None, None]
    a23 = A[..., 1, 2][:, :, None, None]

    u_fish = a11 * x_img + a12 * y_img + a13     # col-pixel
    v_fish = a21 * x_img + a22 * y_img + a23     # row-pixel

    # Validity: in front of camera, inside FOV cone, inside image rect.
    theta_inc = torch.atan2(norm, -Z)            # inclination from forward
    in_front = Z < 0
    in_cone = theta_inc < theta_max

    W = image_wh[..., 0][:, :, None, None]
    H = image_wh[..., 1][:, :, None, None]
    in_image = (u_fish >= 0) & (u_fish < W) & (v_fish >= 0) & (v_fish < H)

    valid = in_front & in_cone & in_image

    u_norm = u_fish / W.clamp(min=1.0)
    v_norm = v_fish / H.clamp(min=1.0)
    u_norm = torch.where(valid, u_norm, torch.full_like(u_norm, 2.0))
    v_norm = torch.where(valid, v_norm, torch.full_like(v_norm, 2.0))

    return torch.stack([u_norm, v_norm], dim=-1)


def kb_project_points(
    key_points: torch.Tensor,
    metas: dict,
    image_wh: torch.Tensor,
) -> torch.Tensor:
    """Project 3D points (lidar frame) with OpenCV Kannala-Brandt fisheye.

    The KB convention used by the KB20 MFTracker dataset is:
        camera frame X=right, Y=down, Z=forward(+)
        K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        D = [k1, k2, k3, k4]

    Unlike a pinhole camera, points with Z <= 0 can still be visible for
    fisheye lenses wider than 180 degrees. Visibility is therefore gated by
    incidence angle and image bounds, not by a hard Z>0 cutoff.
    """
    lidar2cam = metas["lidar2cam"]               # (B, N_cam, 4, 4)
    K = metas.get("kb_intrinsic", metas.get("ocam_intrinsic"))
    D = metas.get("kb_distortion", metas.get("ocam_pol"))
    if K is None or D is None:
        raise KeyError(
            "kb_fisheye projection needs metas['kb_intrinsic'] and "
            "metas['kb_distortion']"
        )

    lens_fov_raw = metas.get(
        "kb_lens_fov_deg", metas.get("ocam_lens_fov_deg", 360.0)
    )
    if isinstance(lens_fov_raw, torch.Tensor):
        lens_fov_deg = float(lens_fov_raw.flatten()[0].item())
    else:
        lens_fov_deg = float(lens_fov_raw)
    theta_max = math.radians(lens_fov_deg / 2.0)

    B = key_points.shape[0]
    N_cam = lidar2cam.shape[1]
    if key_points.dim() == 5:
        if key_points.shape[1] != N_cam:
            raise ValueError(
                "camera-conditioned key_points camera dimension must match "
                f"lidar2cam: {key_points.shape[1]} vs {N_cam}"
            )
        N_anc, num_pts = key_points.shape[2:4]
        ones = key_points.new_ones(B, N_cam, N_anc, num_pts, 1)
        pts_h = torch.cat([key_points, ones], dim=-1)[..., None]
        P_cam = torch.matmul(lidar2cam[:, :, None, None], pts_h).squeeze(-1)
    else:
        N_anc, num_pts = key_points.shape[1:3]
        ones = key_points.new_ones(B, N_anc, num_pts, 1)
        pts_h = torch.cat([key_points, ones], dim=-1)
        pts_h = pts_h[:, None, ..., None]
        P_cam = torch.matmul(lidar2cam[:, :, None, None], pts_h).squeeze(-1)
    X = P_cam[..., 0]
    Y = P_cam[..., 1]
    Z = P_cam[..., 2]

    r = torch.sqrt(X * X + Y * Y + 1e-12)
    theta = torch.atan2(r, Z)

    D = D[..., :4]
    k1 = D[..., 0][:, :, None, None]
    k2 = D[..., 1][:, :, None, None]
    k3 = D[..., 2][:, :, None, None]
    k4 = D[..., 3][:, :, None, None]
    t2 = theta * theta
    theta_d = theta * (
        1.0 + k1 * t2 + k2 * t2 ** 2 + k3 * t2 ** 3 + k4 * t2 ** 4
    )

    scale = torch.where(
        r > 1e-8,
        theta_d / r.clamp(min=1e-8),
        torch.zeros_like(theta_d),
    )
    fx = K[..., 0, 0][:, :, None, None]
    fy = K[..., 1, 1][:, :, None, None]
    cx = K[..., 0, 2][:, :, None, None]
    cy = K[..., 1, 2][:, :, None, None]

    u_fish = fx * X * scale + cx
    v_fish = fy * Y * scale + cy

    W = image_wh[..., 0][:, :, None, None]
    H = image_wh[..., 1][:, :, None, None]
    valid = (
        (theta < theta_max)
        & (u_fish >= 0) & (u_fish < W)
        & (v_fish >= 0) & (v_fish < H)
    )

    u_norm = u_fish / W.clamp(min=1.0)
    v_norm = v_fish / H.clamp(min=1.0)
    u_norm = torch.where(valid, u_norm, torch.full_like(u_norm, 2.0))
    v_norm = torch.where(valid, v_norm, torch.full_like(v_norm, 2.0))
    return torch.stack([u_norm, v_norm], dim=-1)

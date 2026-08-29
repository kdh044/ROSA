import os
"""DAPETR head for KITTI-360's mixed pinhole + MEI fisheye rig.

``DAPETRFisheyeHead`` builds one ``OcamCamera`` and shares its ray grid across
every camera, which is correct for Fisheye3DOD (four identical OCam fisheyes)
and wrong here: KITTI-360 pairs two rectified pinhole cameras with two MEI
fisheyes that do not share a camera model, let alone intrinsics.

Only the geometry is per-camera in this subclass.  The FiLM adaptation, the
positional encoder, the transformer and every loss are inherited untouched, so
this stays a camera-model port of DAPETR rather than a different method.

The parent class is imported, never edited: ``dependencies/fisheye3dod`` is
shared with the Fisheye3DOD experiments and a change there would silently move
those numbers.
"""

from __future__ import annotations

import numpy as np
import torch
from mmdet.models.layers.transformer import inverse_sigmoid
from mmdet3d.registry import MODELS

from fisheye3dod.models.petr.dapetr_head import DAPETRFisheyeHead, pos2posemb3d

from .constants import CAMERAS
from .mixed_geometry import MixedCameraGeometryMixin


@MODELS.register_module()
class DAPETRMixedCameraHead(MixedCameraGeometryMixin, DAPETRFisheyeHead):
    """DAPETR with one camera model per camera instead of one shared OCam."""

    # The parent constructor builds a working ``OcamCamera`` unconditionally
    # (it also derives ``frustum``/``sphere_grid_3d`` from it) before this
    # subclass gets a chance to discard it below, so it needs *some* real
    # calibration file to parse.  Any valid one will do -- its values are
    # never read -- so this points at the Fisheye3DOD calibration already
    # vendored in this workspace rather than asking every KITTI-360 config to
    # know about an OCam implementation detail that is irrelevant to it.
    _DUMMY_OCAM_PATH = (
        os.path.join(os.environ.get("ROSA_DATA_ROOT", "data/Fisheye3DOD"), "calib_results.txt")
    )
    _DUMMY_OCAM_FOV = 220.0

    def __init__(
        self,
        camera_calibration,
        ocam_path: str = _DUMMY_OCAM_PATH,
        ocam_fov: float = _DUMMY_OCAM_FOV,
        **kwargs,
    ) -> None:
        """``camera_calibration`` maps channel -> dict(intrinsic, distortion)."""
        # Set before ``super().__init__`` because the parent constructor calls
        # ``_make_native_geometry`` to fill its cached ray buffers.
        self._camera_calibration = self._validate_calibration(
            camera_calibration
        )
        super().__init__(ocam_path=ocam_path, ocam_fov=ocam_fov, **kwargs)
        # The parent's OCam-derived buffers describe a camera this model never
        # reads (DAPETR's forward builds its own masks) and would otherwise
        # sit in the state dict advertising the wrong geometry.
        for name in ("sphere_grid_2d", "unvalid_area"):
            if hasattr(self, name):
                delattr(self, name)
        self.ocam = None

    def native_position_embedding(self, img_feats, img_metas, masks):
        """Copy of the parent method with ray lifting made per-camera.

        The parent's ``rays`` is ``[H,W,3]`` (one lens shared by all four
        cameras), so its einsum ``'bnij,whdj->bnwhdi'`` broadcasts the same
        camera-frame ray points to every camera index ``n`` and only lets
        the extrinsic ``cam2lidars[b,n]`` vary per camera.  That is exactly
        backwards for a mixed rig: two cameras are pinhole and two are MEI
        fisheye with distinct intrinsics, so the ray direction at a given
        pixel already differs by camera before any extrinsic is applied.
        Here ``rays`` carries a leading camera axis (``[C,H,W,3]``) and the
        einsum gains a matching ``n`` index on the coordinate operand so
        camera ``n``'s points only ever combine with camera ``n``'s rays and
        camera ``n``'s extrinsic.
        """
        batch_size, num_cams, _, height, width = (
            img_feats[self.position_level].shape)
        device = img_feats[self.position_level].device
        dtype = img_feats[self.position_level].dtype
        rays, _ = self._native_geometry(height, width, device=device, dtype=dtype)
        if tuple(rays.shape) != (num_cams, height, width, 3):
            raise ValueError(
                f"expected per-camera rays {(num_cams, height, width, 3)}, "
                f"got {tuple(rays.shape)}"
            )

        index = torch.arange(self.depth_num, device=device, dtype=dtype)
        if self.LID:
            bin_size = (
                self.position_range[3] - self.depth_start
            ) / (self.depth_num * (1 + self.depth_num))
            depth = self.depth_start + bin_size * index * (index + 1)
        else:
            bin_size = (
                self.position_range[3] - self.depth_start
            ) / self.depth_num
            depth = self.depth_start + bin_size * index
        depth_num = depth.numel()

        # [C, W, H, D, 3] in each camera's own local coordinates.
        coords_camera = (
            rays.permute(0, 2, 1, 3).unsqueeze(3)
            * depth.view(1, 1, 1, depth_num, 1)
        )
        coords_camera = torch.cat(
            (coords_camera, torch.ones_like(coords_camera[..., :1])),
            dim=-1,
        )

        cam2lidars = np.asarray([
            img_meta[self.input_key]['cam2lidar']
            for img_meta in img_metas
        ])
        cam2lidars = coords_camera.new_tensor(cam2lidars)
        coords3d = torch.einsum(
            'bnij,nwhdj->bnwhdi', cam2lidars, coords_camera)[..., :3]

        for axis in range(3):
            coords3d[..., axis] = (
                coords3d[..., axis] - self.position_range[axis]
            ) / (
                self.position_range[axis + 3]
                - self.position_range[axis]
            )

        coords_mask = (coords3d > 1.0) | (coords3d < 0.0)
        coords_mask = (
            coords_mask.flatten(-2).sum(-1) > (depth_num * 0.5))
        coords_mask = masks | coords_mask.permute(0, 1, 3, 2)

        coords3d = coords3d.permute(
            0, 1, 4, 5, 3, 2).contiguous().view(
                batch_size * num_cams, -1, height, width)
        coords3d = inverse_sigmoid(coords3d)
        position = self.position_encoder(coords3d).view(
            batch_size, num_cams, self.embed_dims, height, width)

        # DAPETR's three spatial conditions, derived from calibrated rays.
        # rays is [C,H,W,3]; conditions/ray_coords need [B,C,3,H,W].
        incidence = rays[..., 2]
        azimuth = torch.atan2(rays[..., 1], rays[..., 0])
        radial = torch.sqrt(rays[..., 0]**2 + rays[..., 1]**2)
        conditions = torch.stack((incidence, azimuth, radial), dim=1)
        conditions = conditions.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
        ray_coords = rays.permute(0, 3, 1, 2)
        ray_coords = ray_coords.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
        return position, conditions, ray_coords, coords_mask

    def forward(self, mlvl_feats, img_metas):
        """Copy of ``DAPETRFisheyeHead.forward`` with one line fixed.

        The parent builds ``masks`` from a single ``[H,W]`` validity grid
        shared by every camera (``invalid.view(1, 1, H, W).expand(B, C, ...)``)
        because Fisheye3DOD's four cameras share one OCam model.  Here
        ``invalid`` is genuinely per-camera (``[C,H,W]``, pinhole cameras have
        no invalid pixels, the two fisheyes do), so that reshape is wrong --
        it would either raise (element count mismatch) or silently broadcast
        one camera's mask onto all four.  Everything downstream of ``masks``
        is unchanged from the parent; only this construction differs, so the
        method body is copied rather than patched via a hook to keep it
        inspectable as one unit.
        """
        x = mlvl_feats[self.position_level]
        batch_size, num_cams, _, height, width = x.shape
        if num_cams != len(CAMERAS):
            raise ValueError(f"expected {len(CAMERAS)} cameras, got {num_cams}")

        _, invalid = self._native_geometry(
            height, width, device=x.device, dtype=x.dtype
        )
        if tuple(invalid.shape) != (num_cams, height, width):
            raise ValueError(
                f"expected per-camera invalid mask {(num_cams, height, width)}, "
                f"got {tuple(invalid.shape)}"
            )
        masks = invalid.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()

        x = self.input_proj(x.flatten(0, 1)).view(
            batch_size, num_cams, self.embed_dims, height, width)

        if self.with_position:
            position, conditions, ray_coords, _ = self.native_position_embedding(
                mlvl_feats, img_metas, masks
            )
            flat_conditions = conditions.flatten(0, 1)
            x = self.film(
                x.flatten(0, 1),
                flat_conditions[:, 0:1],
                flat_conditions[:, 1:2],
                flat_conditions[:, 2:3],
            ).view_as(x)
            position = self.fpe(
                position.flatten(0, 1), x.flatten(0, 1)).view_as(x)
            pos_embed = position
            if self.with_multiview:
                sine = self.positional_encoding(
                    masks,
                    ray_coords if self.with_polar_pe else None,
                )
                sine = self.adapt_pos3d(sine.flatten(0, 1)).view_as(x)
                pos_embed = pos_embed + sine
            else:
                sine = torch.cat([
                    self.positional_encoding(masks[:, camera]).unsqueeze(1)
                    for camera in range(num_cams)
                ], dim=1)
                pos_embed = pos_embed + self.adapt_pos3d(
                    sine.flatten(0, 1)).view_as(x)
        elif self.with_multiview:
            pos_embed = self.adapt_pos3d(
                self.positional_encoding(masks).flatten(0, 1)).view_as(x)
        else:
            pos_embed = torch.cat([
                self.positional_encoding(masks[:, camera]).unsqueeze(1)
                for camera in range(num_cams)
            ], dim=1)

        reference_points = self.reference_points.weight
        query_embeds = self.query_embedding(pos2posemb3d(reference_points))
        reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)
        outs_dec, _ = self.transformer(
            x, masks, query_embeds, pos_embed, self.reg_branches)
        outs_dec = torch.nan_to_num(outs_dec)

        outputs_classes = []
        outputs_coords = []
        for level in range(outs_dec.shape[0]):
            reference = inverse_sigmoid(reference_points.clone())
            outputs_class = self.cls_branches[level](outs_dec[level]).float()
            output_coord = self.reg_branches[level](outs_dec[level]).float()
            output_coord[..., 0:2] = (
                output_coord[..., 0:2] + reference[..., 0:2]
            ).sigmoid()
            output_coord[..., 4:5] = (
                output_coord[..., 4:5] + reference[..., 2:3]
            ).sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(output_coord)

        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_preds = torch.stack(outputs_coords)
        all_bbox_preds[..., 0:1] = (
            all_bbox_preds[..., 0:1]
            * (self.pc_range[3] - self.pc_range[0])
            + self.pc_range[0]
        )
        all_bbox_preds[..., 1:2] = (
            all_bbox_preds[..., 1:2]
            * (self.pc_range[4] - self.pc_range[1])
            + self.pc_range[1]
        )
        all_bbox_preds[..., 4:5] = (
            all_bbox_preds[..., 4:5]
            * (self.pc_range[5] - self.pc_range[2])
            + self.pc_range[2]
        )
        return {
            'all_cls_scores': all_cls_scores,
            'all_bbox_preds': all_bbox_preds,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
        }

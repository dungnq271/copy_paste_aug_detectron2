import os
import cv2
import random
import numpy as np
import albumentations as A
from copy import deepcopy
from skimage.filters import gaussian
import functional as F


def image_copy_paste(img, paste_img, alpha, blend=True, sigma=1):
    if alpha is not None:
        if blend:
            alpha = gaussian(alpha, sigma=sigma, preserve_range=True)

        img_dtype = img.dtype
        alpha = alpha[..., None]
        img = paste_img * alpha + img * (1 - alpha)
        img = img.astype(img_dtype)

    return img


def background_copy_paste(img, mask, bg_img, height, width, h_start, w_start):
    bg_img = F.random_crop(
        bg_img,
        crop_height=height,
        crop_width=width,
        h_start=h_start,
        w_start=w_start,
    )
    img_dtype = img.dtype
    obj_mask = (mask == 0)[..., None].repeat(3, axis=-1)
    bg_masked = bg_img * obj_mask
    img = bg_masked + img * (~obj_mask)
    img = img.astype(img_dtype)
    return img


def mask_copy_paste(mask, paste_mask, alpha):
    raise NotImplementedError


def masks_copy_paste(masks, paste_masks, alpha):
    if alpha is not None:
        # eliminate pixels that will be pasted over
        masks = [
            np.logical_and(mask, np.logical_xor(mask, alpha)).astype(np.uint8)
            for mask in masks
        ]
        masks.extend(paste_masks)

    return masks


def extract_bboxes(masks):
    bboxes = []
    h, w = masks[0].shape
    for mask in masks:
        yindices = np.where(np.any(mask, axis=0))[0]
        xindices = np.where(np.any(mask, axis=1))[0]
        if yindices.shape[0]:
            y1, y2 = yindices[[0, -1]]
            x1, x2 = xindices[[0, -1]]
            y2 += 1
            x2 += 1
            y1 /= w
            y2 /= w
            x1 /= h
            x2 /= h
        else:
            y1, x1, y2, x2 = 0, 0, 0, 0

        bboxes.append((y1, x1, y2, x2))

    return bboxes


def bboxes_copy_paste(bboxes, paste_bboxes, masks, paste_masks, alpha, key):
    if key == "paste_bboxes":
        return bboxes
    elif paste_bboxes is not None:
        masks = masks_copy_paste(masks, paste_masks=[], alpha=alpha)
        adjusted_bboxes = extract_bboxes(masks)

        # only keep the bounding boxes for objects listed in bboxes
        mask_indices = [box[-1] for box in bboxes]
        adjusted_bboxes = [adjusted_bboxes[idx] for idx in mask_indices]
        # append bbox tails (classes, etc.)
        adjusted_bboxes = [
            bbox + tail[4:] for bbox, tail in zip(adjusted_bboxes, bboxes)
        ]

        # adjust paste_bboxes mask indices to avoid overlap
        if len(masks) > 0:
            max_mask_index = len(masks)
        else:
            max_mask_index = 0

        paste_mask_indices = [
            max_mask_index + ix for ix in range(len(paste_bboxes))
        ]
        paste_bboxes = [
            pbox[:-1] + (pmi,)
            for pbox, pmi in zip(paste_bboxes, paste_mask_indices)
        ]
        adjusted_paste_bboxes = extract_bboxes(paste_masks)
        adjusted_paste_bboxes = [
            apbox + tail[4:]
            for apbox, tail in zip(adjusted_paste_bboxes, paste_bboxes)
        ]

        bboxes = adjusted_bboxes + adjusted_paste_bboxes

    return bboxes


def keypoints_copy_paste(keypoints, paste_keypoints, alpha):
    # remove occluded keypoints
    if alpha is not None:
        visible_keypoints = []
        for kp in keypoints:
            x, y = kp[:2]
            tail = kp[2:]
            if alpha[int(y), int(x)] == 0:
                visible_keypoints.append(kp)

        keypoints = visible_keypoints + paste_keypoints

    return keypoints


class CopyPaste(A.DualTransform):
    def __init__(self, alpha=1.0, sigma=3, p=0.5, always_apply=False):
        super(CopyPaste, self).__init__(always_apply, p)
        self.blend = blend
        self.sigma = sigma
        self.pct_objects_paste = pct_objects_paste
        self.max_paste_objects = max_paste_objects
        self.p = p
        self.always_apply = always_apply

    @staticmethod
    def get_class_fullname():
        return "copypaste.CopyPaste"

    @property
    def targets_as_params(self):
        return [
            "masks",
            "paste_image",
            # "paste_mask",
            "paste_masks",
            "paste_bboxes",
            # "paste_keypoints"
        ]

    def get_params_dependent_on_targets(self, params):
        image = params["paste_image"]
        masks = None
        if "paste_mask" in params:
            # handle a single segmentation mask with
            # multiple targets
            # nothing for now.
            raise NotImplementedError
        elif "paste_masks" in params:
            masks = params["paste_masks"]

        assert masks is not None, "Masks cannot be None!"

        bboxes = params.get("paste_bboxes", None)
        keypoints = params.get("paste_keypoints", None)

        # number of objects: n_bboxes <= n_masks because of automatic removal
        n_objects = len(bboxes) if bboxes is not None else len(masks)

        # paste all objects if no restrictions
        n_select = n_objects
        if self.pct_objects_paste:
            n_select = int(n_select * self.pct_objects_paste)
        if self.max_paste_objects:
            n_select = min(n_select, self.max_paste_objects)

        # no objects condition
        if n_select == 0:
            return {
                "param_masks": params["masks"],
                "paste_img": None,
                "alpha": None,
                "paste_mask": None,
                "paste_masks": None,
                "paste_bboxes": None,
                "paste_keypoints": None,
                "objs_to_paste": [],
            }

        # select objects
        objs_to_paste = np.random.choice(
            range(0, n_objects), size=n_select, replace=False
        )

        # take the bboxes
        if bboxes:
            bboxes = [bboxes[idx] for idx in objs_to_paste]
            # the last label in bboxes is the index of corresponding mask
            mask_indices = [bbox[-1] for bbox in bboxes]

        # create alpha by combining all the objects into
        # a single binary mask
        masks = [masks[idx] for idx in mask_indices]

        alpha = masks[0] > 0
        for mask in masks[1:]:
            alpha += mask > 0

        return {
            "param_masks": params["masks"],
            "paste_img": image,
            "alpha": alpha,
            "paste_mask": None,
            "paste_masks": masks,
            "paste_bboxes": bboxes,
            "paste_keypoints": keypoints,
        }

    @property
    def ignore_kwargs(self):
        return ["paste_image", "paste_mask", "paste_masks"]

    def apply_with_params(
        self, params, force_apply=False, **kwargs
    ):  # skipcq: PYL-W0613
        if params is None:
            return kwargs
        params = self.update_params(params, **kwargs)
        res = {}
        for key, arg in kwargs.items():
            if arg is not None and key not in self.ignore_kwargs:
                target_function = self._get_target_function(key)
                target_dependencies = {
                    k: kwargs[k] for k in self.target_dependence.get(key, [])
                }
                target_dependencies["key"] = key
                res[key] = target_function(
                    arg, **dict(params, **target_dependencies)
                )
            else:
                res[key] = None
        return res

    def apply(self, img, paste_img, alpha, **params):
        return image_copy_paste(
            img, paste_img, alpha, blend=self.blend, sigma=self.sigma
        )

    def apply_to_mask(self, mask, paste_mask, alpha, **params):
        return mask_copy_paste(mask, paste_mask, alpha)

    def apply_to_masks(self, masks, paste_masks, alpha, **params):
        return masks_copy_paste(masks, paste_masks, alpha)

    def apply_to_bboxes(
        self,
        bboxes,
        paste_bboxes,
        param_masks,
        paste_masks,
        alpha,
        key,
        **params
    ):
        return bboxes_copy_paste(
            bboxes, paste_bboxes, param_masks, paste_masks, alpha, key
        )

    def apply_to_keypoints(self, keypoints, paste_keypoints, alpha, **params):
        raise NotImplementedError
        # return keypoints_copy_paste(keypoints, paste_keypoints, alpha)

    def get_transform_init_args_names(self):
        return ("blend", "sigma", "pct_objects_paste", "max_paste_objects")


class ChangeBackground(A.DualTransform):
    def __init__(
        self,
        height: int,
        width: int,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super(ChangeBackground, self).__init__(always_apply, p)
        self.height = height
        self.width = width

    @staticmethod
    def get_class_fullname():
        return "copypaste.ChangeBackground"

    @property
    def targets_as_params(self):
        return [
            "mask",
            "paste_image",
        ]

    def get_params_dependent_on_targets(self, params):
        return {
            "h_start": random.random(),
            "w_start": random.random(),
            "param_mask": params["mask"],
            "paste_img": params["paste_image"]
        }

    def apply(self, img, param_mask, paste_img, h_start, w_start, **params):
        return background_copy_paste(
            img, param_mask, paste_img, self.height, self.width, h_start, w_start
        )

    def apply_to_mask(self, mask, **params):
        return mask

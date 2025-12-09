from typing import Tuple
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import random
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageEnhance, ImageFilter

# ------------------------------
# Your custom augmentations
# ------------------------------
class ElasticTransform:
    def __init__(self, alpha=1, sigma=50, alpha_affine=50, random_state=None):
        self.alpha = alpha
        self.sigma = sigma
        self.alpha_affine = alpha_affine
        self.random_state = random_state

    def __call__(self, img):
        if self.random_state is None:
            random_state = np.random.RandomState(None)
        else:
            random_state = self.random_state

        shape = img.size
        shape_size = shape[:2]

        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32([
            center_square + square_size,
            [center_square[0]+square_size, center_square[1]-square_size],
            center_square - square_size
        ])
        pts2 = pts1 + random_state.uniform(-self.alpha_affine, self.alpha_affine, size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        img = cv2.warpAffine(np.array(img), M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

        dx = gaussian_filter((random_state.rand(*shape_size) * 2 - 1), self.sigma) * self.alpha
        dy = gaussian_filter((random_state.rand(*shape_size) * 2 - 1), self.sigma) * self.alpha

        x, y = np.meshgrid(np.arange(shape_size[1]), np.arange(shape_size[0]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

        remapped = cv2.remap(img, indices[1].astype(np.float32), indices[0].astype(np.float32),
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        return Image.fromarray(remapped)


class RandomNoise:
    def __init__(self, noise_factor=0.1):
        self.noise_factor = noise_factor

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)
        noise = np.random.normal(0, self.noise_factor * 255, img.shape).astype(np.int16)
        noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)


class RandomBrightnessContrast:
    def __init__(self, brightness_range=0.2, contrast_range=0.2):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def __call__(self, img):
        brightness_factor = 1 + random.uniform(-self.brightness_range, self.brightness_range)
        img = ImageEnhance.Brightness(img).enhance(brightness_factor)
        contrast_factor = 1 + random.uniform(-self.contrast_range, self.contrast_range)
        img = ImageEnhance.Contrast(img).enhance(contrast_factor)
        return img


class RandomPerspective:
    def __init__(self, distortion_scale=0.2, p=0.5):
        self.distortion_scale = distortion_scale
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            width, height = img.size
            half_width, half_height = width // 2, height // 2
            topleft = (random.randint(0, int(self.distortion_scale * half_width)), random.randint(0, int(self.distortion_scale * half_height)))
            topright = (width - random.randint(0, int(self.distortion_scale * half_width)), random.randint(0, int(self.distortion_scale * half_height)))
            bottomleft = (random.randint(0, int(self.distortion_scale * half_width)), height - random.randint(0, int(self.distortion_scale * half_height)))
            bottomright = (width - random.randint(0, int(self.distortion_scale * half_width)), height - random.randint(0, int(self.distortion_scale * half_height)))
            startpoints = [(0,0), (width,0), (0,height), (width,height)]
            endpoints = [topleft, topright, bottomleft, bottomright]
            return F.perspective(img, startpoints, endpoints)
        return img


def preprocess_image(image, denoise: bool = True, threshold: bool = False, preserve_fine_details: bool = True):
    """
    Enhanced preprocessing that preserves fine details for ottaksharas.
    
    Args:
        image: PIL Image or numpy array
        denoise: Apply denoising (gentle for ottaksharas)
        threshold: Apply thresholding (avoid for ottaksharas to preserve virama marks)
        preserve_fine_details: Use gentler processing to preserve small glyphs
    """
    if isinstance(image, Image.Image):
        img_array = np.array(image.convert('L'))
    else:
        img_array = image
    
    # Gentle denoising that preserves edges (important for virama and small marks)
    if denoise:
        if preserve_fine_details:
            # Use very gentle bilateral filter to preserve fine details
            img_array = cv2.bilateralFilter(img_array, 3, 30, 30)  # Reduced parameters
        else:
            img_array = cv2.bilateralFilter(img_array, 5, 50, 50)
    
    # Avoid aggressive thresholding for ottaksharas - it can remove virama marks
    if threshold and not preserve_fine_details:
        img_array = cv2.adaptiveThreshold(img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 11, 2)
    
    return Image.fromarray(img_array)


def build_transforms(image_size: int = 64, grayscale: bool = True, enhanced_preprocessing: bool = True) -> Tuple[T.Compose, T.Compose]:
    """Build transforms with correct ToTensor & RandomErasing order"""

    # --------------------------
    # TRAIN TRANSFORMS
    # --------------------------
    train_list = []

    # Optional enhanced preprocessing (preserve fine details for ottaksharas)
    if enhanced_preprocessing:
        train_list.append(T.Lambda(lambda x: preprocess_image(x, denoise=True, threshold=False, preserve_fine_details=True)))

    if grayscale:
        train_list.append(T.Grayscale(num_output_channels=1))

    # Resize & crop
    train_list.extend([
        T.Resize((image_size + 8, image_size + 8), interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.RandomCrop(image_size, padding=4, padding_mode='reflect'),
        T.RandomApply([T.RandomAffine(degrees=15, translate=(0.1,0.1), scale=(0.85,1.15), shear=8,
                                     fill=0, interpolation=T.InterpolationMode.BICUBIC)], p=0.8),
        T.RandomApply([RandomPerspective(distortion_scale=0.3, p=1.0)], p=0.4),
        T.RandomApply([RandomBrightnessContrast(brightness_range=0.3, contrast_range=0.3)], p=0.6),
        T.RandomApply([RandomNoise(noise_factor=0.05)], p=0.3),
        # Reduced blur probability to preserve fine details
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1,0.8))], p=0.2),  # Reduced from 0.3 and max sigma
        T.RandomApply([T.Lambda(lambda x: x.filter(ImageFilter.SHARPEN) if hasattr(x,'filter') else x)], p=0.3),  # Increased sharpening
    ])

    # --------------------------
    # CRITICAL FIX: ToTensor -> Normalize -> RandomErasing
    # --------------------------
    train_list.append(T.ToTensor())
    if grayscale:
        train_list.append(T.Normalize([0.485], [0.229]))
    else:
        train_list.append(T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]))

    # Random erasing after tensor conversion
    train_list.append(T.RandomErasing(p=0.3, scale=(0.02,0.1), ratio=(0.3,3.3), value=0))

    train_tfms = T.Compose(train_list)

    # --------------------------
    # VALIDATION TRANSFORMS
    # --------------------------
    val_list = []
    if enhanced_preprocessing:
        val_list.append(T.Lambda(lambda x: preprocess_image(x, denoise=True, threshold=False, preserve_fine_details=True)))
    if grayscale:
        val_list.append(T.Grayscale(num_output_channels=1))
    val_list.extend([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
    ])
    if grayscale:
        val_list.append(T.Normalize([0.485], [0.229]))
    else:
        val_list.append(T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]))
    val_tfms = T.Compose(val_list)

    return train_tfms, val_tfms

from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,    
    CenterCrop,    
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion, 
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,    
    RandomGamma    
)


def soft_aug(original_height=128, original_width=128):
    aug = Compose([
        OneOf([RandomSizedCrop(min_max_height=(original_height//4, original_height), 
                               height=original_height, width=original_width, p=0.5),
               PadIfNeeded(min_height=original_height, min_width=original_width, p=0.5)], p=1),    
        VerticalFlip(p=0.5),    
        HorizontalFlip(p=0.5),
        RandomRotate90(p=0.5),
        Transpose(p=0.5)
    ])
    return aug


def medium_aug(original_height=128, original_width=128):
    aug = Compose([
        OneOf([RandomSizedCrop(min_max_height=(original_height//4, original_height),
                               height=original_height, width=original_width, p=0.5),
               PadIfNeeded(min_height=original_height, min_width=original_width, p=0.5)], p=1),    
        VerticalFlip(p=0.5),    
        HorizontalFlip(p=0.5),
        RandomRotate90(p=0.5),
        Transpose(p=0.5),
        OneOf([
            ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            GridDistortion(p=0.5),
            OpticalDistortion(p=1, distort_limit=1, shift_limit=0.5)                  
        ], p=0.8)
    ])
    return aug


def hard_aug(original_height=128, original_width=128): 
    aug = Compose([
        OneOf([RandomSizedCrop(min_max_height=(original_height//4, original_height), 
                               height=original_height, width=original_width, p=0.5),
               PadIfNeeded(min_height=original_height, min_width=original_width, p=0.5)], p=1),    
        VerticalFlip(p=0.5),    
        HorizontalFlip(p=0.5),
        RandomRotate90(p=0.5),
        Transpose(p=0.5),
        OneOf([
            ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            GridDistortion(p=0.5),
            OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)                  
        ], p=0.8),
        CLAHE(p=0.8),
        RandomBrightnessContrast(p=0.8),    
        RandomGamma(p=0.8)])
    return aug

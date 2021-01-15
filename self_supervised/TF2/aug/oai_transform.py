import volumentations as V
import albumentations as A

def get_augmentations_3d(patch_size):
    return V.Compose([
        V.RandomResizedCrop(shape=patch_size, scale_limit=(0.65, 1.3)),
        V.Rotate((-15, 15), (-15, 15), (-15, 15), p=0.2),
        V.Flip(0, p=0.5),
        V.Flip(1, p=0.5),
        V.Flip(2, p=0.5),
        V.ElasticTransform(p=0.2),
        V.RandomRotate90(),
        V.GaussianNoise(),
        V.RandomGamma()
    ])

def get_augmentations_2d(patch_size, is_training):
    if is_training:
        return A.Compose([
            A.RandomResizedCrop(height=patch_size[0], width=patch_size[1],scale=(0.65, 1.3)),
            A.Rotate(limit=15, p=0.2),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ElasticTransform(),
            A.RandomRotate90(),
            # A.GaussNoise(),
            # A.RandomGamma()
        ])
    else:
        return A.Compose([
            A.CenterCrop(height=patch_size[0], width=patch_size[1])
        ])


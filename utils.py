import torch
import torchvision.transforms.v2 as v2_transforms


class Config:
    def __init__(self, dictionary):
        for key in dictionary:
            setattr(self, key, dictionary[key])


def get_random_apply(transforms: list[v2_transforms.Transform], prob=0.5):
    return v2_transforms.RandomApply(torch.nn.ModuleList(transforms), p=prob)


class DataAugmentationDINO:
    def __init__(
        self,
        config: Config,
    ):
        flip_and_color_jitter = v2_transforms.Compose(
            [
                v2_transforms.RandomHorizontalFlip(p=0.5),
                v2_transforms.RandomApply(
                    [
                        v2_transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                v2_transforms.RandomGrayscale(p=0.2),
            ]
        )
        normalize = v2_transforms.Compose(
            [
                v2_transforms.ToTensor(),
                v2_transforms.Normalize(config.dataset_means, config.dataset_stds),
            ]
        )

        blur_kernel_size = int(config.img_size * 0.1)

        # first global crop
        self.global_transfo1 = v2_transforms.Compose(
            [
                v2_transforms.RandomResizedCrop(
                    32,
                    scale=config.global_crops_scale,
                    interpolation=v2_transforms.InterpolationMode.BICUBIC,
                ),
                flip_and_color_jitter,
                v2_transforms.GaussianBlur(blur_kernel_size),
                normalize,
            ]
        )

        # second global crop
        self.global_transfo2 = v2_transforms.Compose(
            [
                v2_transforms.RandomResizedCrop(
                    32,
                    scale=config.global_crops_scale,
                    interpolation=v2_transforms.InterpolationMode.BICUBIC,
                ),
                flip_and_color_jitter,
                get_random_apply(
                    [v2_transforms.GaussianBlur(blur_kernel_size)], prob=0.1
                ),
                v2_transforms.RandomSolarize(128, p=0.2),
                normalize,
            ]
        )

        # transformation for the local small crops
        self.local_crops_number = config.local_crops_number
        self.local_transfo = v2_transforms.Compose(
            [
                v2_transforms.RandomResizedCrop(
                    config.local_crop_size,
                    scale=config.local_crops_scale,
                    interpolation=v2_transforms.InterpolationMode.BICUBIC,
                ),
                flip_and_color_jitter,
                get_random_apply(
                    [v2_transforms.GaussianBlur(blur_kernel_size)], prob=0.5
                ),
                normalize,
            ]
        )

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops

import segmentation_models_pytorch as smp
import torchvision as tv
import torchvision.transforms as transforms

model = smp.Unet(
    encoder_name="resnet18",  
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None,  # raw logits produced without activation function
)

loss_fn = smp.losses.DiceLoss(mode='binary')

transforms = tv.transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])
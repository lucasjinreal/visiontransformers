
import torch

from visiontransformers.formers.vit import ViT_b16


def test():
    model = ViT_b16(
        image_size=512,
        num_classes=10)
    
    print(model)

    a = torch.randn([1, 3, 512, 512])
    o = model(a)
    print(o.shape)


if __name__ == '__main__':
    test()

# Vision Transformers



This lib is a collection of different vision transformers, also with some metric reports experimented on our own datasets. Currently supported transformers are:

- ViT;
- NesT;
- DeiT;
- DENO;



Welcome PR or do any experiments on your down datasets and report metric to us. This page will also report speed of very transformer compares with traditional CNNs.



## Start

Before usage, please take `examples` folder as your reference. A simple usage could be:

```python
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

```

And that's all. You get your first ViT transformer model.



## Roadmap

- [ ] Speed benchmark on every model compares with baseline model such as Resnet50;
- [ ] Add training result and metric report on some wild dataset;
- [ ] Add some dense prediction task result such as semantic segmentation / depth estimation.
- [ ] Keep tracking **new breakthroughs** in transformers direction.




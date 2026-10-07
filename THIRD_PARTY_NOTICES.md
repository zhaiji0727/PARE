# Third-party resources

This repository does not vendor dependency packages, pretrained weights, or
SARLANG-1M image/caption assets. The links below identify resources used by
the preparation/runtime tools. Consult their repositories and model/data cards
for licenses, usage conditions and citation requirements; this repository does
not replace or expand those terms.

- [SARLANG-1M dataset](https://huggingface.co/datasets/YiminJimmy/SARLANG-1M)
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [RemoteCLIP](https://github.com/ChenDelong1999/RemoteCLIP) and [weights](https://huggingface.co/chendelong/RemoteCLIP)
- [GeoRSCLIP / RS5M weights](https://huggingface.co/Zilun/GeoRSCLIP)
- [SigLIP model and tokenizer](https://huggingface.co/timm/ViT-B-16-SigLIP)
- [PyTorch](https://github.com/pytorch/pytorch), [torchvision](https://github.com/pytorch/vision)
- [NumPy](https://github.com/numpy/numpy), [SciPy](https://github.com/scipy/scipy), [Pillow](https://github.com/python-pillow/Pillow)
- [timm](https://github.com/huggingface/pytorch-image-models), [Transformers](https://github.com/huggingface/transformers), [Hugging Face Hub](https://github.com/huggingface/huggingface_hub)

The complete dependency/version inventory is in `requirements.lock.txt`.
The archived pre-repair extractor is included only as the comparison reference
for the numerical diagnostic; it is not the default experiment implementation.
No repository-wide license is asserted in this release.

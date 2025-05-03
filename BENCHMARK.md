# Who is the fastest on the market?
We have conducted a fair benchmark of several augmentation libraries by 
comparing how many images they process per second. In this benchmark, we measured
the transform itself, as well as the conversion to `torch.Tensor`, and 
a subtraction of the ImageNet mean. 

Here is how you can run the benchmark yourself on a validation set from ImageNet resized to `256x256x`:

```
export DATA_DIR="<PATH to ImageNet val>"
conda env create -f benchmark/augbench.yaml
conda activate augbench
pip install -U git+https://github.com/imedslab/solt.git
pip install -e benchmark
python -u -m augbench.benchmark -i 500 -r 20  --deterministic --markdown 
```

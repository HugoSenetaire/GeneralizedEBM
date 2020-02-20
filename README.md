# KALE


All runs should be run with:

`python run_gan.py --config=<path_to_configs_file>`


There are three general modes:

1. `train`: Train your own discriminator using the KALE objective. Example configs files in `configs/train`.
2. `eval`: Evaluate the FID scores of a saved discriminator model (automatically saved from step 1) over many examples; examples in `configs/eval`.
3. `fids`: Evaluate the FID scores over fewer examples, but over the course of LMC sampling; examples in `configs/fids`.


### Train

Training will produce a `log_<run_id>.txt` file with the stdout of the run, a `params_<run_id>.txt` with the parameters set in the run, checkpoints in `checkpoints/` with trained discriminator (`d_#.pth`) and/or generator (`g_#.path`) weights, and both prior/posterior samples in `samples/`.

### Eval

Eval runs should include paths to a trained discriminator model (e.g. from the training step). Besides `log_<run_id>.txt` and `params_<run_id>.txt`, this run will produce statistics for KALE and [train, test] statistics for FID after a set amount of steps in `kales_and_fids.json`.


### FIDs

FID runs should include paths to a trained discriminator model (e.g. from the training step). Besides `log_<run_id>.txt` and `params_<run_id>.txt`, this run will produce `posterior_fids.json` with FID scores from over the course of LMC sampling, starting with the same prior. Sample images are included in `samples/`, and GIFs can be made.



--


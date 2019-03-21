# HydroGAN - Replicating n-body simulations using Generative Adversarial Networks

Code acompaining paper: 'HydroGAN - Replicating n-body simulations using Generative Adversarial Networks'


### Prerequisites

* PyTorch 0.4.1
* h5py

### Installing

* Clone this repo to your local machine using
 git clone https://github.com/jjzamudio/Illustris_GAN.git

## Usage

To run WGAN in 3D:

```
python wgan/wgan.py [-h] [--datapath ] 
               [--n_samples N_SAMPLES] [--s_sample S_SAMPLE] [--workers WORKERS]
 

Arguments:

  --datapath DATAPATH                 path to data
  --data-dir DATA_DIR                 path to data directory (used if different from "data/")


```


## References

* [Wasserstein GAN](https://github.com/martinarjovsky/WassersteinGAN)
* [Wasserstein GAN with gradient penalty (https://github.com/EmilienDupont/wgan-gp)




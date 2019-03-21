# HydroGAN - Replicating n-body simulations using Generative Adversarial Networks

Code acompaining paper: 'HydroGAN - Replicating n-body simulations using Generative Adversarial Networks'


### Prerequisites

* PyTorch 0.4.1
* h5py

### Installing

* Clone this repo to your local machine using
 git clone https://github.com/jjzamudio/Illustris_GAN.git

##Usage

To run WGAN in 3D:

```
python wgan/wgan.py [-h] [--datapath DATAPATH] 
               [--plots-dir PLOTS_DIR] [--logs-dir LOGS_DIR]
               [--checkpoints-dir CHECKPOINTS_DIR]
               [--embeddings-dir EMBEDDINGS_DIR] [--dataset-type DATASET_TYPE]
               [--dataset DATASET] [--data-ext DATA_EXT] [--offline] [--force]
               [--cpu] [--cuda] [--device DEVICE] [--device-ids DEVICE_IDS]
               [--parallel] [--emb-model EMB_MODEL]
               [--load-ckpt LOAD_CHECKPOINT] [--load-emb-ckpt LOAD_EMB_CKPT]
               [--load-cls-ckpt LOAD_CLS_CKPT] [--batch-size BATCH_SIZE]
               [--epochs EPOCHS] [--lr LR] [--flatten] [--num-train NUM_TRAIN]
               [--num-frames NUM_FRAMES_IN_STACK]
               [--num-channels NUM_CHANNELS]
               [--num-pairs NUM_PAIRS_PER_EXAMPLE] [--use-pool] [--use-res]


optional arguments:

  --datapath DATAPATH                 path to data
  --data-dir DATA_DIR                 path to data directory (used if different from "data/")
  --plots-dir PLOTS_DIR               path to plots directory (used if different from "logs"plots/)
  --logs-dir LOGS_DIR                 path to logs directory (used if different from "logs/")
  --checkpoints-dir CHECKPOINTS_DIR   path to checkpoints directory (used if different from "checkpoints/")
  --embeddings-dir EMBEDDINGS_DIR     path to embeddings directory (used if different from "checkpoints/embeddings/")
  --dataset-type DATASET_TYPE         name of PyTorch Dataset to use
                                      maze | fixed_mmnist | random_mmnist, default=maze
  --dataset DATASET                   name of dataset file in "data" directory
                                      mnist_test_seq | moving_bars_20_121 | etc., default=all_mazes_16_3_6
  --data-ext DATA_EXT                 extension of dataset file in data directory
  --offline                           use offline preprocessing of data loader
  --force                             overwrites all existing dumped data sets (if used with `--offline`)
  --cpu                               use CPU
  --cuda                              use CUDA, default id: 0
  --device                            cuda | cpu, default=cuda
                                      device to train on
  --device-ids DEVICE_IDS             IDs of GPUs to use
  --parallel                          use all GPUs available
  --emb-model EMB_MODEL               name of embedding network
  --load-ckpt LOAD_CHECKPOINT         name of checkpoint file to load
  --load-emb-ckpt LOAD_EMB_CKPT       name of embedding network file to load
  --load-cls-ckpt LOAD_CLS_CKPT       name of classification network file to load
  --batch-size BATCH_SIZE             input batch size, default=64
  --epochs EPOCHS                     number of epochs, default=10
  --lr LR                             learning rate, default=1e-4
  --flatten                           flatten data into 1 long video
  --num-train NUM_TRAIN               number of training examples
  --num-frames NUM_FRAMES_IN_STACK    number of stacked frames, default=2
  --num-channels NUM_CHANNELS         number of channels in input image, default=1
  --num-pairs NUM_PAIRS_PER_EXAMPLE   number of pairs per video, default=5
  --use-pool                          use max pooling instead of strided convolutions
  --use-res                           use residual layers

```


## References

* [Wasserstein GAN](https://github.com/martinarjovsky/WassersteinGAN)
* [Wasserstein GAN with gradient penalty (https://github.com/EmilienDupont/wgan-gp)




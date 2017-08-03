# pytorch-nips2017-defense-example

This is a baseline, null defense that works within the Cleverhans (https://github.com/tensorflow/cleverhans) framework for the NIPS-2017 adversarial competition. It is intended to be equivalent to the 'base_inception_model' in the sample_defenses but using PyTorch instead of Tensorflow. 

To run:
1. Setup and verify cleverhans nips17 adversarial competition example environment
2. Clone this repo
3. Run ./download_checkpoint.sh to download the inceptionv3 checkpoint from torchvision model zoo
4. Symbolic link the folder this repo was clone into into the cleverhans 'examples/nips17_adversarial_competition/sample_defenses/' folder
5. Run run_attacks_and_defenses.sh but ensure '--gpu' flag is added


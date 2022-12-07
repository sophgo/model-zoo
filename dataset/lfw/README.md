
# lfw dataset

The lfw dataset is used to test facent, and it has been aligned using MTCNN. The pairs.txt stores the sampling results of the dataset. More details please refer [Validate-on-lfw](https://github.com/davidsandberg/facenet/wiki/Validate-on-lfw).

Alignment of the LFW dataset is done something like this:

for N in {1..4}; do \
python src/align/align_dataset_mtcnn.py \
~/datasets/lfw/raw \
~/datasets/lfw/lfw_mtcnnpy_160 \
--image_size 160 \
--margin 32 \
--random_order \
--gpu_memory_fraction 0.25 \
& done

The parameter margin controls how much wider aligned image should be cropped compared to the bounding box given by the face detector. 32 pixels with an image size of 160 pixels corresponds to a margin of 44 pixels with an image size of 182, which is the image size that has been used for training of the model below.
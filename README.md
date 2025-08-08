# FGVC-Aircraft Family-Level Classification with Jetson-Inference

This project demonstrates how to train a fine-grained aircraft family classifier using the [FGVC-Aircraft dataset](https://www.kaggle.com/datasets/seryouxblaster764/fgvc-aircraft) and NVIDIA's [Jetson-Inference](https://github.com/dusty-nv/jetson-inference) framework on a Jetson device.

## Step 1: Download the Dataset

We downloaded the dataset from Kaggle:

https://www.kaggle.com/datasets/seryouxblaster764/fgvc-aircraft  
(Also available from the original source: https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)

The dataset includes image files and label files for training, validation, and test splits. However, this format is not compatible with Jetson-Inference, which expects images organized into folders by class.

## Step 2: Convert Dataset to Jetson-Compatible Format

We used a Python script (`prepare_fgvc_split.py`) to read the training and validation `.txt` files and copy images into subfolders named after each aircraft family.

To run the script:

1. Update the `BASE_DIR` and `OUTPUT_DIR` variables in the script to use your full absolute paths.  
2. Run the script:

    python3 prepare_fgvc_split.py

This organizes the dataset into `train/` and `val/` folders by family name, which is compatible with Jetson-Inference.

## Step 3: Re-Split into 80/20 Train/Val

To get a better training ratio, we merged the existing `train/` and `val/` data and then re-split the images into 80% train and 20% val using a second script (`resplit_train_val.py`).

Run the script:

    python3 resplit_train_val.py

This creates `train_new/` and `val_new/` folders. Once validated, we renamed them:

    mv train train_old  
    mv val val_old  
    mv train_new train  
    mv val_new val

## Step 4: Train the Model

We trained the model using the Jetson-Inference training pipeline.

1. Launch the Docker container:

    cd ~/jetson-inference  
    ./docker/run.sh

2. Navigate to the classification training directory:

    cd python/training/classification

3. Start training:

    python3 train.py --model-dir=models/fgvc_aircraft data/fgvc-aircraft

## Step 5: Export Model to ONNX

After training is complete, convert the model to ONNX format:

    python3 onnx_export.py --model-dir=models/fgvc_aircraft

This generates `resnet18.onnx` and `labels.txt` inside the model directory.

## Step 6: Test the Model

Run inference using an image from the validation set:

    imagenet.py --model=models/fgvc_aircraft/resnet18.onnx --labels=data/fgvc-aircraft/labels.txt --input_blob=input_0 --output_blob=output_0 data/fgvc-aircraft/val/Boeing 747/1234567.jpg output.jpg

This saves the output image with the predicted label overlaid.

## Summary

**Step 1**: Downloaded dataset from [Kaggle](https://www.kaggle.com/datasets/seryouxblaster764/fgvc-aircraft)  
**Step 2**: Converted dataset to folder-per-class format using `prepare_fgvc_split.py`  
**Step 3**: Re-split into 80/20 using `resplit_train_val.py`  
**Step 4**: Trained model using Jetson-Inference  
**Step 5**: Exported model to ONNX format  
**Step 6**: Tested the model with `imagenet.py`

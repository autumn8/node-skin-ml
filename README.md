# Using tensor flow to train a cancer prediction model.

This repo requires a data folder with images populated from [kaggle](https://www.kaggle.com/fanconic/skin-cancer-malignant-vs-benign/data). Download the kaggle data set and unzip it into this repo.

## Prepare data
The folder structure in the kaggle dataset is:
```
> data
   > train
      > benign
      > malignant
   > test
      > benign
      > malignant
    
```
We want the following structure:

```
> data
   > prepared
      > train   
      > test      
```



Benign and malignant images need to be saved to the same folder (train/test) with the lesion type suffixed to the file name:
_0 indicates a benign lesion.
_1 indicates a malignant lesion

Eg. a benign lesion would have a filename such as 1_0.jpg while a malignant lesion might have a filename like 34_1.jpg

Run prepareData to copy images to the prepared folder within the kaggle data folder. The arguments the command line utility accept are the **image data directory** and the **number of images** to copy across for use in your dataset.

## Train, evaluate & save model
``` bash
node prepareData --dir=data --numImages=50
```

## Predict




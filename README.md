Real-time Age and Gender Classifier with TensorFlow 1.8.0
used for POSCO-AI-Academy Project
==========================================================
Referred to Rude Carnie: Age and Gender Deep Learning with TensorFlow
https://github.com/dpressel/rude-carnie

## Goal

Do face detection and age and gender classification on pictures in real time (etc. Web-Cam)

### Currently Supported Models

  - _Gil Levi and Tal Hassner, Age and Gender Classification Using Convolutional Neural Networks, IEEE Workshop on Analysis and Modeling of Faces and Gestures (AMFG), at the IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), Boston, June 2015_

    - http://www.openu.ac.il/home/hassner/projects/cnn_agegender/
    - https://github.com/GilLevi/AgeGenderDeepLearning

  - Inception v3 with fine-tuning
    - This will start with an inception v3 checkpoint, and fine-tune for either age or gender detection

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live webcam system.

It might be used for classifications around those BUS stations, Department stores, ... 

### Data Used

* AFAD

  https://github.com/afad-dataset?tab=overview&from=2015-12-01&to=2015-12-31

* Because the previous model is basically for Western Faces, we have made a new model trained with Asian Faces.  

### Prerequisites

```
pip install -r requirements.txt
```

### Preprocessing

```
PreproHelpers/AFAD_prepro.ipynb
```

```
PreproHelpers/create_train_val_txt_files.ipynb
```

Then directory 'folds' will appear on --outputdir and 
text files like :
age_train.txt, age_val.txt, age_test.txt, ...
gender_train.txt, gender_train.txt, ...

should be made in fold_dir 

This means that you are ready to preprocess for training 
```
$ python preproc.py --fold_dir /home/dpressel/dev/work/AgeGenderDeepLearning/Folds/train_val_txt_files_per_fold/test_fold_is_0 --train_list age_train.txt --valid_list age_val.txt --data_dir /data/xdata/age-gender/aligned --output_dir /home/dpressel/dev/work/AgeGenderDeepLearning/Folds/tf/age_test_fold_is_0

```
Now you are 100% ready!! Those image Data & Text Data labeled Age&Gender are successfully preprocessed.
  
#### Train the model (Levi/Hassner)

Now that we have generated the training and validation shards, we can start training the program.  Here is a simple way to call the driver program to run using SGD with momentum to train:

```
$ python train.py --train_dir /home/dpressel/dev/work/AgeGenderDeepLearning/Folds/tf/age_test_fold_is_0

```

Once again, gender is done much the same way.  Just be careful that you are running on the the preprocessed gender data, not the age data.  Here we use a lower initial learning rate of `0.001`

```

$ python train.py --train_dir /home/dpressel/dev/work/AgeGenderDeepLearning/Folds/tf/gen_test_fold_is_0 --max_steps 30000 --eta 0.001

```

## Acknowledgments

* Rude Carnie (https://github.com/dpressel/rude-carnie)
* AFAD (https://github.com/afad-dataset?tab=overview&from=2015-12-01&to=2015-12-31)

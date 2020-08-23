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
python version should be 3.6.x

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

#### Train the model (fine-tuned Inception)

Its also easy to use this codebase to fine-tune an pre-trained inception checkpoint for age or gender dectection.  Here is an example for how to do this:

```
$ python train.py --train_dir /home/dpressel/dev/work/AgeGenderDeepLearning/Folds/tf/age_test_fold_is_0 --max_steps 15000 --model_type inception --batch_size 32 --eta 0.001 --dropout 0.5 --pre_model /data/pre-trained/inception_v3.ckpt
```

You can get the inception_v3.ckpt like so:

```
$ wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
```

#### Get your model ready ####

make directories for both age and gender model!
ex) age_model/, gender_model/
And move yours to the folders 

#### Run Live Video Streaming #### 

```
$ python livestreaming_age_gender_optimized.py (run.py)

```
![result](https://eastasia1-mediap.svc.ms/transform/thumbnail?provider=spo&inputFormat=png&cs=NWUzY2U2YzAtMmIxZi00Mjg1LThkNGItNzVlZTc4Nzg3MzQ2fFNQTw&docid=https%3A%2F%2Fsookmyungackr.sharepoint.com%3A443%2Fsites%2FB3AIBigdata%2F_api%2Fv2.0%2Fdrives%2Fb!a-S9g3E66kWcickRATYN1dm43Z2WdA5JlRdzVIznrn4fXBEi0TlmRYAvmL2oJNEb%2Fitems%2F01RKR4GM7S7BZZVEXHUVDIN4DUARWR5Q2D%3Fversion%3DPublished&access_token=eyJ0eXAiOiJKV1QiLCJhbGciOiJub25lIn0.eyJhdWQiOiIwMDAwMDAwMy0wMDAwLTBmZjEtY2UwMC0wMDAwMDAwMDAwMDAvc29va215dW5nYWNrci5zaGFyZXBvaW50LmNvbUA5NzE5YjBhNi0yYWFlLTRjNDQtOTIzNS1lMmRmZDczYTMxZWMiLCJpc3MiOiIwMDAwMDAwMy0wMDAwLTBmZjEtY2UwMC0wMDAwMDAwMDAwMDAiLCJuYmYiOiIxNTk4MTkzNjIzIiwiZXhwIjoiMTU5ODIxNTIyMyIsImVuZHBvaW50dXJsIjoieHVGVHUxei9YdlNZbUZLcnBVdGlWSHYvN2drTFY2b0NMZnNxK2tUY0xOVT0iLCJlbmRwb2ludHVybExlbmd0aCI6IjEzOCIsImlzbG9vcGJhY2siOiJUcnVlIiwiY2lkIjoiTUROa1pqY3lPV1l0WmpBM09TMHdNREF3TFRVNU4yWXROREJsTWpZMU1tVTROVFZoIiwidmVyIjoiaGFzaGVkcHJvb2Z0b2tlbiIsInNpdGVpZCI6Ik9ETmlaR1UwTm1JdE0yRTNNUzAwTldWaExUbGpPRGt0WXpreE1UQXhNell3WkdRMSIsImFwcF9kaXNwbGF5bmFtZSI6Ik1pY3Jvc29mdCBUZWFtcyBXZWIgQ2xpZW50Iiwic2lnbmluX3N0YXRlIjoiW1wia21zaVwiXSIsImFwcGlkIjoiNWUzY2U2YzAtMmIxZi00Mjg1LThkNGItNzVlZTc4Nzg3MzQ2IiwidGlkIjoiOTcxOWIwYTYtMmFhZS00YzQ0LTkyMzUtZTJkZmQ3M2EzMWVjIiwidXBuIjoiMjAxNDEwNDEzNl9raHUuYWMua3IjZXh0I0Bzb29rbXl1bmdhY2tyLm9ubWljcm9zb2Z0LmNvbSIsInB1aWQiOiIxMDAzMjAwMENDODg4MEFDIiwiY2FjaGVrZXkiOiIwaC5mfG1lbWJlcnNoaXB8MTAwMzIwMDBjYzg4ODBhY0BsaXZlLmNvbSIsInNjcCI6Im15ZmlsZXMud3JpdGUgYWxsc2l0ZXMuZnVsbGNvbnRyb2wgYWxsc2l0ZXMubWFuYWdlIGFsbHByb2ZpbGVzLndyaXRlIiwidHQiOiIyIiwidXNlUGVyc2lzdGVudENvb2tpZSI6bnVsbH0.YmRSMisvcXJ2NzVtK2Z3Q0NaR0dUWjFZNXFmVkZDYytPd0tBc1BVRWltRT0&encodeFailures=1&width=1224&height=692)

## Acknowledgments

* Rude Carnie (https://github.com/dpressel/rude-carnie)
* AFAD (https://github.com/afad-dataset?tab=overview&from=2015-12-01&to=2015-12-31)

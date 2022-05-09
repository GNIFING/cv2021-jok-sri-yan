# 2110433 Computer Vision กลุ่มโจ๊กศรีย่าน

This repository is for the the 2110433 Computer Vision term project. The project outlines pose classification problem to classify captured real-time video in order to control Subway Surfer-styled Game.

## Prerequisites

Please find the `requirements.txt` in top directory to install relevant libralies. This project uses `Python 3.9.12`

```bash
pip install -r requirements.txt
```

If python 3.9 is not available, you can install these libraries manually:
* numpy
* mediapipe
* opencv
* pynput
* Flask

Any python version >= 3.7 should suffice (not tested), but should not be over than 3.10 (python < 3.10) as `Pynput` is not supported in python 3.10.

For `.ipynb` in `/other`, if run locally, additional libraries are also required:
* pandas
* matplotlib

## Project Directory

```txt
cv2021-jok-sri-yan/
├─ script/
│  └─ pose_estimator.py             // Main Control
├─ web/
│  ├─ app.py                        // Web Server
│  └─ ...
├─ other/
|  ├─ mediapipe_gen_csv.ipynb       // Landmark Dataset Generation
|  ├─ ml_classifier.ipynb           // ML Models Training and Evaluation
|  ├─ rule_classifier_eval.ipynb    // Rule-based System Evaluation
|  └─ landmarks_fixed.csv           // Landmark Dataset
└─ presentation_slide.pdf           // Presentation Slide (Thai)
```

This repository consists of 3 folders:
* `script/` stores the main control flow task for pose classification and control on the game. This includes pose estimation, rule-based classification, and game input controls.
* `web/` stores all relevant files for the web server. `app.py` serves as the entrypoint.
* `other` stores Jupyter notebooks which were used in the development process. This includes ML training pipeline (dataset csv generation from MediaPipe landmarks to train models and the model training notebook) and rule-based classifier evaluation, which uses the same csv as the ML model training.

## Startup

### Main Pose Classification and Control

1. Please ensure that the desired camera to use is the camera number `0`. If not, please change the `CAM_NO` constant on line `7` in `script/pose_estimator.py`.

2. Start the script by
```bash
python script/pose_estimator.py
```

3. If only pose classification task is needed without the keyboard control, please change the `CONTROL_KEYBOARD` on line `8` in the same file `script/pose_estimator.py` to `False`

Note that the description of the program is provided on the website. Moreover, the technical aspect of the program is as presented on the presentation slide.

### Web Server

To initialize the web server, change the current directory to `/web` and start the server.
```bash
cd /web
flask run
```

If this OSError is emitted (`OSError: [WinError 10013] An attempt was made to access a socket in a way forbidden by its access permissions`), please use this command to start the server instead of `flask run`.
```bash
python app.py runserver
```

If the server is deployed to remote host, the Demo tab might not work correctly as the demo section uses the server's camera (in case of localhost server, it is your own computer's camera) from OpenCV library. `CAM_NO` in `web/app.py` can also be changed like the main script for game control.

### Notebooks in `other/` folder

Notebooks in `other/` are downloaded from Google Colab. Uploading and running on Google Colab is more recommended as we did not change the script to work locally with Jupyter.

Only landmark dataset (already converted by MediaPipe) is provided as `landmarks_fixed.csv` which is used to train and evaluate ML models. The original images are not included in this repository (Please contact us if needed).

## Discussions

### Problem Statement

This project aims to provide and alternative method to play computer game: to use player's posing as game inputs. The whole pipeline consists of 3 parts:

1. Pose Estimation (MediaPipe)
2. Pose Classification (Rule-based)
3. Game Control (Pynput through keyboard keys)

The whole pipeline takes real-time images as input and give prediction as key control as the output. We also aim that the classifier should not depend on the position of the player, meaning that before-playing calibration should not required at all.

### Technical Challenges, Methods and Results

#### Pose Estimation: State-of-the-art models

The pose estimation part uses a deep learning model to predict body landmarks. OpenPose, MediaPipe, and MoveNet were tested; it was found that MediaPipe is the most suitable for this work. OpenPose could not process fast enough (might be due to some misconfigurations though) and MoveNet is not quite good in term of accuracy.

#### Pose Classification: Traditional ML vs Rule-based

Both traditional machine learning models and rule-based system were experimented.

ML: We have experimented with XGBoost and KNN, both with and without landmarks' translational normalization. The results were not good at first, but we have figured it was a fault in training script; we programmed the script to read the folders of each pose images and forgot to clear the list of images before moving to the next folder, therefore the labeling was chaotic. After fixing, the results show a 100% accuracy with the test set (which might be a sign of too small dataset).

Rule-based: We have explored the conditions for each of the 4 poses: Raising the left hand, Raising the right hand, Bending the knees, and Jumping. The left and right hands raising were easy enough, checking only if these landmarks' `y` coordinates are above the shoulder's `y`. For bending the knees, standard deviation of lower body landmarks' `y` are computed and should be less than a certain threshold. However, it came tricky for the jumping, as classifying a single frame of jumping might be impossible. Jumping looks like standing still if only a single frame is provided. Thus, we compared the `y` coordinate of some upper body landmarks that they should be more than their moving average of `y` to a certain threshold to trigger the jumping condition. For both bending knees and jumping, we also normalized the condition by the width of the player's shoulder, as we aim that the pose prediction should be independent of the player location from the camera (thus, the system does not need further calibration). The result from evaluation with the same dataset from ML training shows that it is capable enough (6 images were misclassified, 5 out of which is compensable by human eyes).

The selection of which to use is another matter as both shows high accuracy towards the evaluation dataset. Although, ML models are at a 100% accuracy which might suggest that the dataset is not big enough. Moreover, a rule base system is unargurably faster and more interpretable than a machine learning model. Thus, we decided to use the rule-based model for our system.

#### Game Control: Converting predictions to real control

This part does not prove much difficulty as we decided to control the game through keyboard inputs (Pynput already provides the interface). One-shot mechanism was implemented to convert the real-time prediction into keyboard control, much like in Digital Circuit where only a single pulse of signal is output from a One-Shot logic gate which connected to an input button.

### Related Works

Some of these works we have found after implementing this project can solve our same use case to a certain similarity. As we did not find them earlier, our project might or might not employ the same mechanism to these works, in which we are unsure; but they are worth further studying anyway.

* Subway Surfers Gesture Control: https://www.youtube.com/watch?app=desktop&v=W3fkJF7SLjk&feature=youtu.be
* Playing NeedForSpeed By Hand Gesture Control: https://create.arduino.cc/projecthub/najad/playing-nfs-by-hand-gesture-control-ba3d8a

### Future Work

Some ideas were suggested as follow:

* Rule-based Classifier: As we only consider `y` coordinates to compute the standard deviation to predict knee bending pose, additionally considering `x` coordinates might help better accuracy.
* Rule-based Classifier: All poses except for the jumping pose are predicted from a static frame. If we figured rules to consider past frames much like how we implemented jumping detection, the control might appear more fluid.
* UX: Multi-class classification might seem more interesting, so players can input multiple control at the same time.
* UX: More poses to consider.
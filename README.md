# Hand-Gesture-Classification-Based-on-EMG-signals

The aim of the project was to use multiple machine learning models to recognize the hand gestures based on 8-channel EMG signals for 36 subjects. Hand gesture recognition can be beneficial in many practical applications, for example, monitoring the recovery process of patients, translating gestures to human languages, human-computer interaction, to name just a few. Generally speaking, two major approaches for hand gesture recognition are available, including vision-based approaches and bio-signal-based approaches. Bio-signal-based approaches are more efficient in terms of data and computation, and what is more important is that bio-signal-based approaches directly links myographic signals with hand gesture, revealing how signals brain sends to arm control hand gestures. The challenge is that myographic signals collected with sensors sometimes can be noisy and less interpretable. 

The data consists of signals collected from MYO Thalmic bracelets worn on users’ forearms. The bracelet is equipped with eight sensors equally spaced around the forearm that simultaneously acquire myographic signals. The raw EMG data for 36 subjects are recorded while they performed series of static hand gestures. The subjects perform two series, each of which consists of six (seven) basic gestures. Each gesture was performed for 3 seconds with a pause of 3 seconds between gestures.

This was a two-person group project for a machine learning course. The project consisted of two main phases. The data preprocessing and denoising stages handled by my partner, and the feature selection and machine learning model development which was handled by me. Initially, we performed the cleaning and filtering processes on the raw EMG data to improve the data quality. Thereafter, we adopted overlapping sliding window for data segmentation and extracted five time-domain features of each channel. 

We then removed redundancy by reducing highly correlated features. Using these features, we split the data into 75% train size and 25% test size. Following this, we built classifiers with support vector machine (SVM), logistic regression (LR), random forest (RF) and multilayer perceptron (MLP), then tuned the hyperparameters to find the best models. 

From the classification models we used, the best model was determined to be MLP with an accuracy of 98.756%, followed by the SVM, then the RF and finally the LR model. Finally, we ensembled the best models of these four models using the stacking technique. This combined model using the stacking ensemble technique yielded the best accuracy which was 0.249% higher compared to that of the MLP model.

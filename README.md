# music-genre-classification
Music genre classification project- learning ML techniques

Created a neural network and a convolutional nueral network for classifying songs into 10 different genres from the GTZAN dataset: https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification 

### Inspiration
For this project, my goal was to learn more about how machine learning works. Some of the files that I have included have to do with implementing a nueral network. The main files of this project are genre_classifier.py and cnn_genre_classfier.py. I chose to predict the genre of music because I find it interesting that something as complex as a song can be broken down into numerical data that machine learning techniques can take advantage of! 

### Future Plans 
This project is working and predicts the genre pretty accurately (more details below) but in the future, I'd like to add some other ML techniques to this repo to practice, and maybe make this more functional for a user. 


### GTZAN Dataset attributes:
- 10 genres with 100 files each, all with a length of 30 seconds
- A visual representation of each sound
- 2 CSV files containing the features of the audio files 

Original datset compiled by Andrada Olteanu on Kaggle.com (and other sources)

## The project 

First step was to preprocess the dataset. To do this, I implemented code to batch process the dataset in order to extract the MFCCs (mel frequency cepstral coefficients) and genre labels. MFCCs are a small set of features that describe the overall shape of a spectral envelope for a song. Then, we can use a JSON file to make it easy to use this data in our NN, CNN, etc. 

Then, a multiclass classifier was created using Tensorflow, trained on the dataset. The model performed extremely well on the training data (~96% accuracy), but performed poorly on the test data (~60% accruacy). So, I had to solve the overfitting problem.

#### Solving overfitting:

Here, we can see the overfitting issue.

![Overfitting](/plot1)

We can see that the test accuracy doesn't quite reach the same accuracy as the train accuracy. It stales at about 60%, so there is a huge difference in the test and train accuracy. For the error, we have something similar happening. To fix this, we need the data to generalize with data it has never seen before. To solve this, I decided to use dropout and regularization (and kind of augmenting the data) using TensorFlow. 

First, I augmented the data, but not in a typical way. Sometimes, it is common for audio data to be augmented by shifting the pithc, time stretching, and adding background noise. This gives the network more training data and deviations from "perfect" data so that the algorithm performs better on data it hasn't seen yet. However, I decided not to implement this. Instead, I split up the 30 second songs into smaller pieces to artificially give the network more data to train with. 

Early stopping prevents the network from overfitting before it happens- as the network sees more of the same data, it gets less accurate when it sees data that isn't exactly what it saw it training. In the plot above, if we stop at the 10th epoch, the difference between the train/test sets in accuracy and error becomes smaller than it is after it trains with all the epochs. 

Dropout randomly drops nuerons during training and incresed network robustness because it has to figure out a way to get around problems that arise when nuerons are dropped. It trains "around" the missing nueron and relies less on any specific nueron using this method. There is no universal rule for drouput rates, but I went with a value between 0.5 - 0.10. 

Here is the plot after implenting data augmentation, droput, and early stopping. 


![Overfitting-Fixed](/plot2)

The accuracy/error between the train and test sets is much less now!








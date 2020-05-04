---
layout: post
title: Real-time music annotation with deep learning in Essentia
image: /assets/tensorflow-real-time/tensorflow-real-time.png
category: news
---
In this post, we demonstrate how to use TensorFlow models in Essentia for real-time music audio annotation. Find more information about [the available models](https://mtg.github.io/essentia-labs/news/2020/01/16/tensorflow-models-released/) and the introduction to [our TensorFlow wrapper](https://mtg.github.io/essentia-labs/news/2019/10/19/tensorflow-models-in-essentia/) in our previous posts.

Real-time constraints are a common requirement for many applications that involve digital audio signal processing and analysis in a wide variety of contexts, and deployment scenarios, and the cutting edge approaches relying on deep learning should be no exception. For this reason, we have equipped Essentia with algorithms that support inference with deep learning models in real-time.

This real-time capability, however, ultimately relies on the complexity of the models and the computational resources available. Some of our models currently available for download are lightweight enough to perform real-time predictions with a CPU on a regular laptop, and we have prepared a demo to show you this!

We use a pre-trained auto-tagging model based on the MusiCNN architecture introduced in [our previous post](https://mtg.github.io/essentia-labs/news/2019/10/19/tensorflow-models-in-essentia/), that we adapted to operate on shorter one-second chunks of audio. This process did not require any additional training, but it reduced the performance of the model a bit. For predictions, we compute a mel-spectrogram of each audio chunk as an input to this model.

The video below shows a moving window with the mel-spectrograms used for prediction on the left and the resulting tag activations on the right.

<iframe width="600" height="480" src="http://www.youtube.com/embed/t1emx0_U3zw" frameborder="0" allowfullscreen></iframe>

The code can be found in this [notebook](https://github.com/pabloEntropia/mtg-general-meeting-03-2020-essentia-tensorflow/blob/master/demo-realtime-essentia-tensorflow.ipynb).

One cool feature of our collection of models is that the same processing chain can be applied to the transfer learning classifiers based on the MusiCNN architecture. These classifier models were obtaining by retraining the last layer of the MusiCNN auto-tagging model, and therefore we can use all of them simultaneously without any computational overhead. In the next video, we can see the real-time predictions of the classifiers trained for detecting aggressive and happy moods, danceability and voice activity in music.

<iframe width="600" height="480" src="http://www.youtube.com/embed/IWcb8Jx2bk0" frameborder="0" allowfullscreen></iframe>

We envision many promising applications of this technology as the capability to recognize high-level characteristics on music streams or radio broadcasts. In music production, this could be used to generate valuable real-time feedback with high-level descriptors or to develop a new generation of powerful audio effects based on deep-learning.

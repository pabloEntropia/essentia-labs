---
layout: post
title: Real-time deep-learning predictions with Essentia
category: news
---
Due to the temporal nature of sound, real-time constraints are a requirement for many related signal processing applications. Examples of this are the modern encoding algorithms that compress the audio and video streams preventing teleconferences from all around the world from collapsing the internet or the modern mixing consoles allowing the simultaneous use of hundreds of audio effects to improve the quality of live performances.

As a continuation of this idea, modern approaches relying on deep learning must be able to operate under the same constraints to be usable in many deployment scenarios. To this end, we have equipped Essentia with algorithms that support real-time deep-learning inference.

The real-time capability, however, ultimately relies on the complexity of the model and the computational resources available. In this sense, some of our models are compact enough to perform real-time predictions with the CPU of a regular laptop and we have created open demos exemplifying this!

Our demos rely on [SoundCard](https://soundcard.readthedocs.io/en/latest/) to catch the computer audio loopback (whatever is being played from Youtube, Spotify, your local player...) and stream it into Essentia. We are using our auto-tagging model based in the MusiCNN architecture, already introduced on a [previous post](https://mtg.github.io/essentia-labs/news/2019/10/19/tensorflow-models-in-essentia/). We have adapted our model to operate on chunks of one second of audio. This process does not require any extra training, but it reduces the performance of the model a bit. The video below shows a moving window with the mel bands used for prediction on the left and the activated tags on the right.

[VIDEO 1]

The code can be be found in this [notebook](https://github.com/pabloEntropia/mtg-general-meeting-03-2020-essentia-tensorflow/blob/master/demo-realtime-essentia-tensorflow.ipynb).

One cool feature of our collection of models is that the same chain can be applied to the transfer learning classifiers based on the MusiCNN architecture. Given that those models were obtaining by retraining the last layer of the auto-tagging model, we can use all those without any computational overhead. In the next example, we can see some of our transfer learning classifiers operating simultaneously in real-time.

[VIDEO 2]

We envision many promising applications of this technology as the capability to recognize high-level characteristics on music streams or radio broadcasts. In music production, this could be used to generate valuable real-time feedback with high-level descriptors or to develop a new generation of powerful audio effects based on deep-learning.

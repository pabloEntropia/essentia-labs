---
layout: post
title: Real-time music annotation with deep learning in Essentia
image: /assets/tensorflow-real-time/tensorflow-real-time.png
category: news
tags: [TensorFlow]
---
In this post, we demonstrate how to use TensorFlow models in Essentia for real-time music audio annotation. Find more information about [the available models](https://mtg.github.io/essentia-labs/news/2020/01/16/tensorflow-models-released/) and the introduction to [our TensorFlow wrapper](https://mtg.github.io/essentia-labs/news/2019/10/19/tensorflow-models-in-essentia/) in our previous posts.

Real-time constraints are a common requirement for many applications that involve digital audio signal processing and analysis in a wide variety of contexts, and deployment scenarios, and the approaches relying on deep learning should be no exception. For this reason, we have equipped Essentia with algorithms that support inference with deep learning models in real-time.
This real-time capability, however, ultimately relies on the complexity of the models and the computational resources at hand. Some of our models currently available for download are lightweight enough to perform real-time predictions with a CPU on a regular laptop, and we have prepared a demo to show you this!

Our demo utilizes [SoundCard](https://github.com/bastibe/SoundCard) to read the computer audio loopback, capturing all audio that is playing on the system, such as coming from a local music player application or a web browser (e.g., from Youtube). It is streamed to Essentia in real-time for the prediction of music tags that can be associated with the audio.
We use a pre-trained auto-tagging model based on the MusiCNN architecture, introduced in [our previous post](https://mtg.github.io/essentia-labs/news/2019/10/19/tensorflow-models-in-essentia/), adapted to operate on small one-second chunks of audio.
This process does not require any extra training, but it reduces the performance of the model a bit. A mel-spectrogram of each chunk serves as an input to the model.

The video below shows a moving mel-spectrogram window on the left and the resulting tag activations on the right.

<iframe width="600" height="480" src="http://www.youtube.com/embed/t1emx0_U3zw" frameborder="0" allowfullscreen></iframe>

The code can be found in this [notebook](https://github.com/pabloEntropia/mtg-general-meeting-03-2020-essentia-tensorflow/blob/master/demo-realtime-essentia-tensorflow.ipynb).

One cool feature of our collection of models is that the same processing chain can be applied to the transfer learning classifiers based on the MusiCNN architecture. These classifier models were obtaining by retraining the last layer of the MusiCNN auto-tagging model, and therefore we can use all of them simultaneously without any computational overhead. In the next video, you can see the real-time predictions of the classifiers trained for detecting aggressive and happy moods, danceability, and voice activity in music.

<iframe width="600" height="480" src="http://www.youtube.com/embed/IWcb8Jx2bk0" frameborder="0" allowfullscreen></iframe>

You are welcome to try this demo on your own, and of course, it can be adapted to many other TensorFlow models for sound and music audio annotation.

We envision many promising applications of the real-time inference functionality in Essentia. For example, we can use it for recognition of relevant semantic characteristics in the context of online music streaming and radio broadcasts. In music production tools, it can help to generate valuable real-time feedback based on "smarter" audio analysis and foster development of powerful audio effects using deep learning.

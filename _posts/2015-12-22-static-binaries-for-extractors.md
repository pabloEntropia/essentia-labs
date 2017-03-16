---
layout: post
title: Static binaries for command-line extractors for music descriptors
category: news
---

We are now hosting static binaries for a number command-line extractors, originally developed as examples of how Essentia can be used in applications. These tools make it easy to compute some descriptors and store them to a file given an audio file as an input (therefore, called "extractors") without any need to install Essentia library itself.

In particular these extractors can:

- compute a large set of spectral, time-domain, rhythm, tonal and high-level descriptors
- compute MFCC frames
- extract pitch of a predominant melody using MELODIA algorithm
- extract pitch for a monophonic signal using YinFFT algorithm
- extract beats

Notably, two extractors, specifically designed for [AcousticBrainz](http://acousticbrainz.org/) and [Freesound](http://freesound.org/) projects are included. They include large sets of features and are designed for batch processing large amounts of audio files.

See description of the extractors in the [official documentation](http://essentia.upf.edu/documentation/extractors_out_of_box.html). Specifically, [here](http://essentia.upf.edu/documentation/streaming_extractor_music.html) you will find details about our music extractor.

Current builds are done for our latest version of Essentia [2.1_beta2](https://github.com/MTG/essentia/releases/tag/v2.1_beta2) (Linux, OSX, and Windows). Download them here: [http://essentia.upf.edu/documentation/extractors](http://essentia.upf.edu/documentation/extractors).

Of course, you are welcome to submit your own extractors to be included in our future builds.

---
layout: post
title: Fingerprinting with Chromaprint algorithm 
category: news
---

As we are gradually expanding Essentia with new functionality, we have added a new algorithm for computation of audio fingerprints, the [Chromaprinter](http://essentia.upf.edu/documentation/reference/std_Chromaprinter.html). Technically, it is a wrapper of [Chromaprint](https://acoustid.org/chromaprint) library which you will need to install to be able to use Chromaprinter. 

The fingerprints computed with Chromaprinter can be used to query the [AcoustID](https://acoustid.org/) database for track metadata. Check a few examples of how Chromaprinter can be used in Python [here](http://essentia.upf.edu/documentation/essentia_python_examples.html#fingerprinting). To start using this algorithm now, build the latest Essentia code from the [master branch](https://github.com/MTG/essentia/tree/master).

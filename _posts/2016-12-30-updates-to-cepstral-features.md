---
layout: post
title: Updates to cepstral features (MFCC and GFCC) 
---

Working towards the next Essentia release, we have updated our cepstral features. The updates include:

- Support for extracting MFCCs 'the htk way' ([python example](https://github.com/MTG/essentia/blob/master/src/examples/tutorial/example_mfcc_the_htk_way.py)).

- In literature there are two common MFCCs 'standards' differing in some parameters and the mel-scale computation itself: the Slaney way ([Auditory toolbox](https://engineering.purdue.edu/%7Emalcolm/interval/1998-010/)) and the htk way (chapter 5.4 from [htk book](http://www.dsic.upv.es/docs/posgrado/20/RES/materialesDocentes/alejandroViewgraphs/htkbook.pdf)).

- See a [python notebook](https://github.com/georgid/mfcc-htk-an-librosa/blob/master/htk%20and%20librosa%20MFCC%20extract%20comparison.ipynb) for a comparison with mfcc extracted with librosa and with htk.

- Support for inverting the computed MFCCs back to spectral (mel) domain ([python example](https://github.com/MTG/essentia/blob/master/src/examples/tutorial/example_inverse_mfccs.py)).

- The first MFCC coefficients are standard for describing singing voice timbre. The MFCC feature vector however does not represent the singing voice well visually. Instead, it is a common practice to invert the first 12-15 MFCC coefficients back to mel-bands domain for visualization. We have ported invmelfcc.m as explained [here](http://labrosa.ee.columbia.edu/matlab/rastamat/).

- Support for [cent scale](http://essentia.upf.edu/documentation/reference/std_SpectrumToCent.html).

You can start using these features before the official release by building Essentia from the [master branch](https://github.com/MTG/essentia/tree/master).

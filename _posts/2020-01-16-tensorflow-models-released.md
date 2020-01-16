---
layout: post
title: A collection of TensorFlow models for Essentia
category: news
---
We present a collection of TensorFlow models for Essentia.
These models achieve State of The Art results in the task of music auto-tagging. Additionally, we include a set of classifiers trained on our in-house datasets on top of the larger auto-tagging models.

We also provide convenience algorithms that allow performing predictions in just a couple of lines. Also, we include C++ extractors that could be compiled statically to use as standalone executables.

## The models

### Auto-tagging models
These models were trained as part of Jordi Pons' Ph.D. Thesis. More details about the training process can be found in the [original repository](https://github.com/jordipons/musicnn-training).

| Architecture | Dataset | Task | Parameters | Version | Accuracy | Download link |
|---|---|---|---|---|---|---|
| MusiCNN | MSD | auto-tagging | 790k | all |   | [MSD_musicnn_frozen_all](https://essentia.upf.edu/models/autotagging/MSD_musicnn_frozen_all.pb) |
| MusiCNN | MSD | auto-tagging | 790k | small |   | [MSD_musicnn_frozen_small](https://essentia.upf.edu/models/autotagging/MSD_musicnn_frozen_small.pb) |
| MusiCNN | MTT | auto-tagging | 790k | all | | [MSD_vgg_frozen_all](https://essentia.upf.edu/models/autotagging/MTT_musicnn_frozen_all.pb) |
| MusiCNN | MTT | auto-tagging | 790k | small | | [MSD_vgg_frozen_small](https://essentia.upf.edu/models/autotagging/MTT_musicnn_frozen_small.pb) |
| VGG | MSD | auto-tagging | 605k | all | | [MTT_musicnn_frozen_all](https://essentia.upf.edu/models/autotagging/MSD_vgg_frozen_all.pb) |
| VGG | MSD | auto-tagging | 605k | small |   | [MTT_musicnn_frozen_small](https://essentia.upf.edu/models/autotagging/MSD_vgg_frozen_small.pb) |
| VGG | MTT | auto-tagging | 605k | all | | [MTT_vgg_frozen_all](https://essentia.upf.edu/models/autotagging/MTT_vgg_frozen_all.pb) |
| VGG | MTT | auto-tagging | 605k | small | | [MTT_vgg_frozen_small](https://essentia.upf.edu/models/autotagging/MTT_vgg_frozen_small.pb) |

### Transfer Learning classifiers
This is a set of classifiers trained on top of the penalultimate layer of larger CNN auto-taggers.

| Architecture | Source Dataset | Target Dataset | Parameters | Accuracy | Download |
|---|---|---|---|---|---|
| MusiCNN | MSD | danceability | 790k |   |   |
| MusiCNN | MSD | gender | 790k |   |   |
| MusiCNN | MSD | genre_dortmund | 790k |   |   |
| MusiCNN | MSD | genre_electronic | 790k |   |   |
| MusiCNN | MSD | genre_rosamerica | 790k |   |   |
| MusiCNN | MSD | genre_tzanetakis | 790k |   |   |
| MusiCNN | MSD | mood_acoustic | 790k |   |   |
| MusiCNN | MSD | mood_aggressive | 790k |   |   |
| MusiCNN | MSD | mood_electronic | 790k |   |   |
| MusiCNN | MSD | mood_happy | 790k |   |   |
| MusiCNN | MSD | mood_party | 790k |   |   |
| MusiCNN | MSD | mood_relaxed | 790k |   |   |
| MusiCNN | MSD | mood_sad | 790k |   |   |
| MusiCNN | MSD | tonal_atonal | 790k |   |   |
| MusiCNN | MSD | voice_instrumental | 790k |   |   |
| MusiCNN | MTT | danceability | 790k |   |   |
| MusiCNN | MTT | gender | 790k |   |   |
| MusiCNN | MTT | genre_dortmund | 790k |   |   |
| MusiCNN | MTT | genre_electronic | 790k |   |   |
| MusiCNN | MTT | genre_rosamerica | 790k |   |   |
| MusiCNN | MTT | genre_tzanetakis | 790k |   |   |
| MusiCNN | MTT | mood_acoustic | 790k |   |   |
| MusiCNN | MTT | mood_aggressive | 790k |   |   |
| MusiCNN | MTT | mood_electronic | 790k |   |   |
| MusiCNN | MTT | mood_happy | 790k |   |   |
| MusiCNN | MTT | mood_party | 790k |   |   |
| MusiCNN | MTT | mood_relaxed | 790k |   |   |
| MusiCNN | MTT | mood_sad | 790k |   |   |
| MusiCNN | MTT | tonal_atonal | 790k |   |   |
| MusiCNN | MTT | voice_instrumental | 790k |   |   |
| VGGish | Audioset | danceability | 62M |   |   |
| VGGish | Audioset | gender | 62M |   |   |
| VGGish | Audioset | genre_dortmund | 62M |   |   |
| VGGish | Audioset | genre_electronic | 62M |   |   |
| VGGish | Audioset | genre_rosamerica | 62M |   |   |
| VGGish | Audioset | genre_tzanetakis | 62M |   |   |
| VGGish | Audioset | mood_acoustic | 62M |   |   |
| VGGish | Audioset | mood_aggressive | 62M |   |   |
| VGGish | Audioset | mood_electronic | 62M |   |   |
| VGGish | Audioset | mood_happy | 62M |   |   |
| VGGish | Audioset | mood_party | 62M |   |   |
| VGGish | Audioset | mood_relaxed | 62M |   |   |
| VGGish | Audioset | mood_sad | 62M |   |   |
| VGGish | Audioset | tonal_atonal | 62M |   |   |
| VGGish | Audioset | voice_instrumental | 62M |   |   |

### Architecture details
  - [MusiCNN](https://github.com/jordipons/musicnn) is a musically-motivated Convolutional Neural Network. It uses vertical and horizontal convolutional filters aiming to capture timbral and temporal patterns, respectively. The model contains 6 layers and 790k parameters.
  - [VGG]() is an architecture from computer vision based on a deep stack of 3X3 convolutional filters commonly used. It contains 5 layers with 128 filters each. Batch normalization and dropout are applied before each layer. The model has 605k trainable parameters. We are using [Jordi Pon's implementation](https://github.com/jordipons/musicnn)
  - [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset) follows the configuration E from the original implementation for computer vision, with the difference that the number of output units is set to 3087. This model has 62 million parameters.

### Datasets details
  - [MSD](http://millionsongdataset.com/lastfm) contains 200k tracks from the train set of the publicly available Million Song Dataset (MSD) annotated by the 50 Lastfm tags most frequent in the dataset.
  - [MTT](http://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset) contains data collected using the TagATune game and music from the Magnatune label. It contains 25k tracks.
  - [AudioSet](https://research.google.com/audioset/) contains 1.8 million audio clips from Youtube annotated with the AudioSet taxonomy, not specific to music.
  - **MTG in-house datasets** is a collection of small, highly curated datasets used for training classifiers. [A set of SVM classifiers](https://acousticbrainz.org/datasets/accuracy) based on these datset is also available.
  
| Dataset            | Classes                                                                                                                     | Size           |
|--------------------|-----------------------------------------------------------------------------------------------------------------------------|----------------|
| genre\-dortmund    | alternative, blues, electronic, folkcountry, funksoulrnb, jazz, pop, raphiphop, rock                                        | 1820 exc\.     |
| genre\-gtzan       | blues, classic, country, disco, hip hop, jazz, metal, pop, reggae, rock                                                     | 1000 exc\.     |
| genre\-rosamerica  | classic, dance, hip hop, jazz, pop, rhythm and blues, rock, speech                                                          | 400 ft\.       |
| genre\-electronic | ambient, dnb, house, techno, trance                                                                                         | 250 exc\.      |
| mood\-acoustic     | acoustic, not acoustic                                                                                                      | 321 ft\.       |
| mood\-electronic   | electronic, not electronic                                                                                                  | 332 ft\./exc\. |
| mood\-aggressive   | aggressive, not aggressive                                                                                                  | 280 ft\.       |
| mood\-relaxed      | not relaxed, relaxed                                                                                                        | 446 ft\./exc\. |
| mood\-happy        | happy, not happy                                                                                                            | 302 exp\.      |
| mood\-sad          | not sad, sad                                                                                                                | 230 ft\./exc\. |
| mood\-party        | not party, party                                                                                                            | 349 exp\.      |
| danceability       | danceable, not dancable                                                                                                     | 306 ft\.       |
| voice/instrumental | voice, instrumental                                                                                                         | 1000 exc\.     |
| gender             | female, male                                                                                                                | 3311 ft\.      |
| tonal/atonal       | atonal, tonal                                                                                                               | 345 exc\.      |

## The new algorithms
Algorithms for `MusiCNN` and `VGG` based models:
- **TensorflowInputMusiCNN**. Computes mel-bands with a particular parametrization specific
  to MusiCNN based models. 
- **TensorflowPredictMusiCNN**. Makes predictions using MusiCNN models.

Algorithms for `VGGish` based models:
- **TensorflowInputVGGish**. Computes mel-bands with a particular parametrization specific
  to VGGish based models.
- **TensorflowPredictVGGish**. Makes predictions using VGGish models.

## Usage examples
Now that we have a clear idea of which models are available, let's exemplify some use cases.

### Auto-tagging
In this case, we are replicating the behavior on our [previous post](https://mtg.github.io/essentia-labs//news/2019/10/19/tensorflow-models-in-essentia/). With the new algorithms, the code is reduced to just a couple of lines!

```python
import numpy as np
from essentia.standard import *


msd_labels = ['rock','pop','alternative','indie','electronic','female vocalists','dance','00s','alternative rock','jazz','beautiful','metal','chillout','male vocalists','classic rock','soul','indie rock','Mellow','electronica','80s','folk','90s','chill','instrumental','punk','oldies','blues','hard rock','ambient','acoustic','experimental','female vocalist','guitar','Hip-Hop','70s','party','country','easy listening','sexy','catchy','funk','electro','heavy metal','Progressive rock','60s','rnb','indie pop','sad','House','happy']

# Our models take audio streams at 16kHz
sr = 16000

# Instantiate a MonoLoader and run it in the same line
audio = MonoLoader(filename='/your/amazong/song.wav', sampleRate=sr)()

# Instatiate the tagger and pass it the audio
predictions = TensorflowPredictMusiCNN(graphFilename='MSD_musicnn_frozen_ssmall.pb')(audio)

# Retrieve the top_n tags
top_n = 3

# Take advantage of NumPy to average and sort the predictions
for i, l in enumerate(np.mean(predictions, axis=0).argsort()[-top_n:][::-1], 1):
    print('{}: {}'.format(i, msd_labels[l]))

```

```
1: electronic
2: chillout
3: ambient
```

### Classification
In this example, we are using the `rosamerica` classifier based on the VGGish embeddings. Note that we are using `TensorflowPredictVGGish` instead of `TensorflowPredictMusiCNN` so that the model is fed with the correct input features.

```python
import numpy as np
from essentia.standard import *


labels = ['classic', 'dance', 'hip hop', 'jazz', 'pop', 'rnb', 'rock', 'speech']

sr = 16000
audio = MonoLoader(filename='/your/amazong/song.wav', sampleRate=sr)()


predictions = TensorflowPredictVGGish(graphFilename='genre_rosamerica_vggish_audioset.pb')(audio)
predictions = np.mean(predictions, axis=0)
order = predictions.argsort()[::-1]
for i in order:
    print('{}: {:.3f}'.format(labels[i], predictions[i]))
```

```
hip hop: 0.411
dance: 0.397
jazz: 0.056
pop: 0.053
rnb: 0.051
rock: 0.011
classic: 0.004
speech: 0.001

```

### Feature extraction
As the last example, let's use one of the models as a feature extractor by retrieving the output of the penultimate layer. This is done by setting the `output` parameter in the predictor algorithm.
A list of the supported output layer is available on the `README` files [shipped with the models](https://essentia.upf.edu/models/).

```python
from essentia.standard import MonoLoader, TensorflowPredictMusiCNN


sr = 16000
audio = MonoLoader(filename='/your/amazong/song.wav', sampleRate=sr)()

# Retrieve the output of the penultimate layer
penultimate_layer = TensorflowPredictMusiCNN(graphFilename='MSD_musicnn_frozen_small.pb'
                                             output='model/dense_1/BiasAdd')(audio)

```


![png]({{ site.baseurl }}/assets/tensorflow-models-released/penultimate_layer_feats.png)

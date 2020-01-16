---
layout: post
title: A collection of TensorFlow models for Essentia
category: news
---
In our [last post](https://mtg.github.io/essentia-labs//news/2019/10/19/tensorflow-models-in-essentia/), we introduced the TensorFlow wrapper for Essentia. It can be used with virtually any TensorFlow model and here we present a collection of models we supply with Essentia out of box.

First, we prepared a set of pre-trained `auto-tagging models` achieving state of the art performance. Then, we used those models as feature extractors to generate a set of `transfer learning classifiers` trained on our in-house datasets.

Along with the models we provide convenience algorithms that allow performing predictions in just a couple of lines. Also, we include C++ extractors that could be built statically to use as standalone executables.

## Supplied models

### Auto-tagging models
Our current auto-tagging were pre-trained as part of Jordi Pons' [Ph.D. Thesis](https://www.upf.edu/web/mdm-dtic/thesis/-/asset_publisher/vfmxwU8uwTZk/content/-phd-thesis-deep-neural-networks-for-music-and-audio-tagging). They were trained to predict the top 50 tags from [lastFm](https://www.last.fm). Check the [original repository](https://github.com/jordipons/musicnn-training) for more details about the training process.

The following table shows the available architectures and the datasets used for training. For every model, its complexity is reported in terms of the number of trainable parameters. The models were evaluated using [standarized train/test splits](http://millionsongdataset.com/pages/tasks-demos/index.html). The performance of the models is indicated in terms of `ROC-AUC` and `PR-AUC` obtained in the test split as it is common for tagger systems.

| Architecture | Dataset | Params. | ROC-AUC | PR-AUC | Download link |
|---|---|---|---|---|---|
| MusiCNN | MSD | 790k | 0.88 | 0.29 | [MSD_musicnn_frozen_small](https://essentia.upf.edu/models/autotagging/MSD_musicnn_frozen_small.pb) |
| MusiCNN | MTT | 790k | 0.91 | 0.38 | [MSD_vgg_frozen_small](https://essentia.upf.edu/models/autotagging/MTT_musicnn_frozen_small.pb) |
| VGG     | MSD | 605k | 0.88 | 0.28 | [MTT_musicnn_frozen_small](https://essentia.upf.edu/models/autotagging/MSD_vgg_frozen_small.pb) |
| VGG     | MTT | 605k | 0.90 | 0.38 | [MTT_vgg_frozen_small](https://essentia.upf.edu/models/autotagging/MTT_vgg_frozen_small.pb) |

### Transfer learning classifiers
In a transfer learning schema, a model is first trained on a source task (typically with greater amount of available data) to leverage the obtained knowledge in a smaller target task. In this case, we considered the aforementioned MusiCNN models and a big VGG-like (VGGish) model as source tasks. As target tasks, we considered our in-house classification datasets explained below. We use the penultimate layer of the source task models as feature extractors for small classifiers consisting of 2 fully connected layers.


For each classifier, the following table shows its source task as a combination of the architecture and the training dataset used. Its complexity is reported in terms of the number of trainable parameters. The performance of each model is measured in terms of normalized accuracy in a 5-fold cross-validation experiment.

| Source task | Target task | Params. | Norm acc. | Download link |
|---|---|---|---|---|
| MusiCNN-MSD      | danceability       | 790k | 0.93 | [danceability_musicnn_msd](https://essentia.upf.edu/models/transfer_learning_classifiers/danceability/danceability_musicnn_msd.pb) |
| MusiCNN-MSD      | gender             | 790k | 0.88 | [gender_musicnn_msd](https://essentia.upf.edu/models/transfer_learning_classifiers/gender/gender_musicnn_msd.pb) |
| MusiCNN-MSD      | genre_dortmund     | 790k | 0.51 | [genre_dortmund_musicnn_msd](https://essentia.upf.edu/models/transfer_learning_classifiers/genre_dortmund/genre_dortmund_musicnn_msd.pb) |
| MusiCNN-MSD      | genre_electronic   | 790k | 0.95 | [genre_electronic_musicnn_msd](https://essentia.upf.edu/models/transfer_learning_classifiers/genre_electronic/genre_electronic_musicnn_msd.pb) |
| MusiCNN-MSD      | genre_rosamerica   | 790k | 0.92 | [genre_rosamerica_musicnn_msd](https://essentia.upf.edu/models/transfer_learning_classifiers/genre_rosamerica/genre_rosamerica_musicnn_msd.pb) |
| MusiCNN-MSD      | genre_tzanetakis   | 790k | 0.83 | [genre_tzanetakis_musicnn_msd](https://essentia.upf.edu/models/transfer_learning_classifiers/genre_tzanetakis/genre_tzanetakis_musicnn_msd.pb) |
| MusiCNN-MSD      | mood_acoustic      | 790k | 0.90 | [mood_acoustic_musicnn_msd](https://essentia.upf.edu/models/transfer_learning_classifiers/mood_acoustic/mood_acoustic_musicnn_msd.pb) |
| MusiCNN-MSD      | mood_aggressive    | 790k | 0.95 | [mood_aggressive_musicnn_msd](https://essentia.upf.edu/models/transfer_learning_classifiers/mood_aggressive/mood_aggressive_musicnn_msd.pb) |
| MusiCNN-MSD      | mood_electronic    | 790k | 0.95 | [mood_electronic_musicnn_msd](https://essentia.upf.edu/models/transfer_learning_classifiers/mood_electronic/mood_electronic_musicnn_msd.pb) |
| MusiCNN-MSD      | mood_happy         | 790k | 0.81 | [mood_happy_musicnn_msd](https://essentia.upf.edu/models/transfer_learning_classifiers/mood_happy/mood_happy_musicnn_msd.pb) |
| MusiCNN-MSD      | mood_party         | 790k | 0.89 | [mood_party_musicnn_msd](https://essentia.upf.edu/models/transfer_learning_classifiers/mood_party/mood_party_musicnn_msd.pb) |
| MusiCNN-MSD      | mood_relaxed       | 790k | 0.90 | [mood_relaxed_musicnn_msd](https://essentia.upf.edu/models/transfer_learning_classifiers/mood_relaxed/mood_relaxed_musicnn_msd.pb) |
| MusiCNN-MSD      | mood_sad           | 790k | 0.86 | [mood_sad_musicnn_msd](https://essentia.upf.edu/models/transfer_learning_classifiers/mood_sad/mood_sad_musicnn_msd.pb) |
| MusiCNN-MSD      | tonal_atonal       | 790k | 0.60 | [tonal_atonal_musicnn_msd](https://essentia.upf.edu/models/transfer_learning_classifiers/tonal_atonal/tonal_atonal_musicnn_msd.pb) |
| MusiCNN-MSD      | voice_instrumental | 790k | 0.98 | [voice_instrumental_musicnn_msd](https://essentia.upf.edu/models/transfer_learning_classifiers/voice_instrumental/voice_instrumental_musicnn_msd.pb) |
| MusiCNN-MTT      | danceability       | 790k | 0.91 | [danceability_musicnn_mtt](https://essentia.upf.edu/models/transfer_learning_classifiers/danceability/danceability_musicnn_mtt.pb) |
| MusiCNN-MTT      | gender             | 790k | 0.87 | [gender_musicnn_mtt](https://essentia.upf.edu/models/transfer_learning_classifiers/gender/gender_musicnn_mtt.pb) |
| MusiCNN-MTT      | genre_dortmund     | 790k | 0.44 | [genre_dortmund_musicnn_mtt](https://essentia.upf.edu/models/transfer_learning_classifiers/genre_dortmund/genre_dortmund_musicnn_mtt.pb) |
| MusiCNN-MTT      | genre_electronic   | 790k | 0.71 | [genre_electronic_musicnn_mtt](https://essentia.upf.edu/models/transfer_learning_classifiers/genre_electronic/genre_electronic_musicnn_mtt.pb) |
| MusiCNN-MTT      | genre_rosamerica   | 790k | 0.92 | [genre_rosamerica_musicnn_mtt](https://essentia.upf.edu/models/transfer_learning_classifiers/genre_rosamerica/genre_rosamerica_musicnn_mtt.pb) |
| MusiCNN-MTT      | genre_tzanetakis   | 790k | 0.80 | [genre_tzanetakis_musicnn_mtt](https://essentia.upf.edu/models/transfer_learning_classifiers/genre_tzanetakis/genre_tzanetakis_musicnn_mtt.pb) |
| MusiCNN-MTT      | mood_acoustic      | 790k | 0.93 | [mood_acoustic_musicnn_mtt](https://essentia.upf.edu/models/transfer_learning_classifiers/mood_acoustic/mood_acoustic_musicnn_mtt.pb) |
| MusiCNN-MTT      | mood_aggressive    | 790k | 0.96 | [mood_aggressive_musicnn_mtt](https://essentia.upf.edu/models/transfer_learning_classifiers/mood_aggressive/mood_aggressive_musicnn_mtt.pb) |
| MusiCNN-MTT      | mood_electronic    | 790k | 0.91 | [mood_electronic_musicnn_mtt](https://essentia.upf.edu/models/transfer_learning_classifiers/mood_electronic/mood_electronic_musicnn_mtt.pb) |
| MusiCNN-MTT      | mood_happy         | 790k | 0.79 | [mood_happy_musicnn_mtt](https://essentia.upf.edu/models/transfer_learning_classifiers/mood_happy/mood_happy_musicnn_mtt.pb) |
| MusiCNN-MTT      | mood_party         | 790k | 0.92 | [mood_party_musicnn_mtt](https://essentia.upf.edu/models/transfer_learning_classifiers/mood_party/mood_party_musicnn_mtt.pb) |
| MusiCNN-MTT      | mood_relaxed       | 790k | 0.88 | [mood_relaxed_musicnn_mtt](https://essentia.upf.edu/models/transfer_learning_classifiers/mood_relaxed/mood_relaxed_musicnn_mtt.pb) |
| MusiCNN-MTT      | mood_sad           | 790k | 0.85 | [mood_sad_musicnn_mtt](https://essentia.upf.edu/models/transfer_learning_classifiers/mood_sad/mood_sad_musicnn_mtt.pb) |
| MusiCNN-MTT      | tonal_atonal       | 790k | 0.91 | [tonal_atonal_musicnn_mtt](https://essentia.upf.edu/models/transfer_learning_classifiers/tonal_atonal/tonal_atonal_musicnn_mtt.pb) |
| MusiCNN-MTT      | voice_instrumental | 790k | 0.98 | [voice_instrumental_musicnn_mtt](https://essentia.upf.edu/models/transfer_learning_classifiers/voice_instrumental/voice_instrumental_musicnn_mtt.pb) |
| VGGish-AudioSet | danceability        | 62M  | 0.94 | [danceability_vggish_audioset](https://essentia.upf.edu/models/transfer_learning_classifiers/danceability/danceability_vggish_audioset.pb) |
| VGGish-AudioSet | gender              | 62M  | 0.84 | [gender_vggish_audioset](https://essentia.upf.edu/models/transfer_learning_classifiers/gender/gender_vggish_audioset.pb) |
| VGGish-AudioSet | genre_dortmund      | 62M  | 0.52 | [genre_dortmund_vggish_audioset](https://essentia.upf.edu/models/transfer_learning_classifiers/genre_dortmund/genre_dortmund_vggish_audioset.pb) |
| VGGish-AudioSet | genre_electronic    | 62M  | 0.93 | [genre_electronic_vggish_audioset](https://essentia.upf.edu/models/transfer_learning_classifiers/genre_electronic/genre_electronic_vggish_audioset.pb) |
| VGGish-AudioSet | genre_rosamerica    | 62M  | 0.94 | [genre_rosamerica_vggish_audioset](https://essentia.upf.edu/models/transfer_learning_classifiers/genre_rosamerica/genre_rosamerica_vggish_audioset.pb) |
| VGGish-AudioSet | genre_tzanetakis    | 62M  | 0.86 | [genre_tzanetakis_vggish_audioset](https://essentia.upf.edu/models/transfer_learning_classifiers/genre_tzanetakis/genre_tzanetakis_vggish_audioset.pb) |
| VGGish-AudioSet | mood_acoustic       | 62M  | 0.94 | [mood_acoustic_vggish_audioset](https://essentia.upf.edu/models/transfer_learning_classifiers/mood_acoustic/mood_acoustic_vggish_audioset.pb) |
| VGGish-AudioSet | mood_aggressive     | 62M  | 0.98 | [mood_aggressive_vggish_audioset](https://essentia.upf.edu/models/transfer_learning_classifiers/mood_aggressive/mood_aggressive_vggish_audioset.pb) |
| VGGish-AudioSet | mood_electronic     | 62M  | 0.93 | [mood_electronic_vggish_audioset](https://essentia.upf.edu/models/transfer_learning_classifiers/mood_electronic/mood_electronic_vggish_audioset.pb) |
| VGGish-AudioSet | mood_happy          | 62M  | 0.86 | [mood_happy_vggish_audioset](https://essentia.upf.edu/models/transfer_learning_classifiers/mood_happy/mood_happy_vggish_audioset.pb) |
| VGGish-AudioSet | mood_party          | 62M  | 0.91 | [mood_party_vggish_audioset](https://essentia.upf.edu/models/transfer_learning_classifiers/mood_party/mood_party_vggish_audioset.pb) |
| VGGish-AudioSet | mood_relaxed        | 62M  | 0.89 | [mood_relaxed_vggish_audioset](https://essentia.upf.edu/models/transfer_learning_classifiers/mood_relaxed/mood_relaxed_vggish_audioset.pb) |
| VGGish-AudioSet | mood_sad            | 62M  | 0.89 | [mood_sad_vggish_audioset](https://essentia.upf.edu/models/transfer_learning_classifiers/mood_sad/mood_sad_vggish_audioset.pb) |
| VGGish-AudioSet | tonal_atonal        | 62M  | 0.97 | [tonal_atonal_vggish_audioset](https://essentia.upf.edu/models/transfer_learning_classifiers/tonal_atonal/tonal_atonal_vggish_audioset.pb) |
| VGGish-AudioSet | voice_instrumental  | 62M  | 0.98 | [voice_instrumental_vggish_audioset](https://essentia.upf.edu/models/transfer_learning_classifiers/voice_instrumental/voice_instrumental_vggish_audioset.pb) |

### Architecture details
  - [MusiCNN](https://github.com/jordipons/musicnn) is a musically-motivated Convolutional Neural Network. It uses vertical and horizontal convolutional filters aiming to capture timbral and temporal patterns, respectively. The model contains 6 layers and 790k parameters.
  - [VGG]() is an architecture from computer vision based on a deep stack of 3X3 convolutional filters commonly used. It contains 5 layers with 128 filters each. Batch normalization and dropout are applied before each layer. The model has 605k trainable parameters. We are using [Jordi Pon's implementation](https://github.com/jordipons/musicnn)
  - [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset) follows the configuration E from the original implementation for computer vision, with the difference that the number of output units is set to 3087. This model has 62 million parameters.

### Datasets details
  - [MSD](http://millionsongdataset.com/lastfm) contains 200k tracks from the train set of the publicly available Million Song Dataset (MSD) annotated by the 50 Lastfm tags most frequent in the dataset.
  - [MTT](http://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset) contains data collected using the TagATune game and music from the Magnatune label. It contains 25k tracks.
  - [AudioSet](https://research.google.com/audioset/) contains 1.8 million audio clips from Youtube annotated with the AudioSet taxonomy, not specific to music.
  - **MTG in-house datasets** are a collection of small, highly curated datasets used for training classifiers. [A set of SVM classifiers](https://acousticbrainz.org/datasets/accuracy) based on these datsaets is also available.
  
| Dataset            | Classes                                                                                                                     | Size           |
|--------------------|-----------------------------------------------------------------------------------------------------------------------------|----------------|
| genre-dortmund     | alternative, blues, electronic, folkcountry, funksoulrnb, jazz, pop, raphiphop, rock                                        | 1820 exc.      |
| genre-gtzan        | blues, classic, country, disco, hip hop, jazz, metal, pop, reggae, rock                                                     | 1000 exc.      |
| genre-rosamerica   | classic, dance, hip hop, jazz, pop, rhythm and blues, rock, speech                                                          | 400 ft.        |
| genre-electronic   | ambient, dnb, house, techno, trance                                                                                         | 250 exc.       |
| mood-acoustic      | acoustic, not acoustic                                                                                                      | 321 ft.        |
| mood-electronic    | electronic, not electronic                                                                                                  | 332 ft./exc.   |
| mood-aggressive    | aggressive, not aggressive                                                                                                  | 280 ft.        |
| mood-relaxed       | not relaxed, relaxed                                                                                                        | 446 ft./exc.   |
| mood-happy         | happy, not happy                                                                                                            | 302 exp.       |
| mood-sad           | not sad, sad                                                                                                                | 230 ft./exc.   |
| mood-party         | not party, party                                                                                                            | 349 exc.       |
| danceability       | danceable, not dancable                                                                                                     | 306 ft.        |
| voice/instrumental | voice, instrumental                                                                                                         | 1000 exc.      |
| gender             | female, male                                                                                                                | 3311 ft.       |
| tonal/atonal       | atonal, tonal                                                                                                               | 345 exc.       |

## Helper algorithms
Algorithms for `MusiCNN` and `VGG` based models:
- **TensorflowInputMusiCNN**. Computes mel-bands with a particular parametrization specific
  to MusiCNN based models. 
- **TensorflowPredictMusiCNN**. Makes predictions using MusiCNN models.

Algorithms for `VGGish` based models:
- **TensorflowInputVGGish**. Computes mel-bands with a particular parametrization specific
  to VGGish based models.
- **TensorflowPredictVGGish**. Makes predictions using VGGish models.

## Usage examples
Let's exemplify some use cases.

### Auto-tagging
In this case, we are replicating the example of our [previous post](https://mtg.github.io/essentia-labs//news/2019/10/19/tensorflow-models-in-essentia/). With the new algorithms, the code is reduced to just a couple of lines!

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


labels = ['classic', 'dance', 'hip hop', 'jazz',
          'pop', 'rnb', 'rock', 'speech']

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

### Latent feature extraction
As the last example, let's use one of the models as a feature extractor by retrieving the output of the penultimate layer. This is done by setting the `output` parameter in the predictor algorithm.
A list of the supported output layers is available in the `README` files [supplied with the models](https://essentia.upf.edu/models/).

```python
from essentia.standard import MonoLoader, TensorflowPredictMusiCNN


sr = 16000
audio = MonoLoader(filename='/your/amazong/song.wav', sampleRate=sr)()

# Retrieve the output of the penultimate layer
penultimate_layer = TensorflowPredictMusiCNN(graphFilename='MSD_musicnn_frozen_small.pb',
  output='model/dense_1/BiasAdd')(audio)

```
The following plot shows how these features look like:

![png]({{ site.baseurl }}/assets/tensorflow-models-released/penultimate_layer_feats.png)

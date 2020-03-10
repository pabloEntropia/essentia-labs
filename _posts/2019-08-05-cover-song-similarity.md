---
layout: post
title: Cover song similarity algorithms in Essentia
category: news
---


Cover song identification or Version Identification is a task of identifying when two musical recordings are derived from the same music composition. A cover version of a song can be drastically different from the original recording. It may have variations in key, tempo, structure, melody, harmony, timbre, language and lyrics compared to its original version, which makes it a challenging task [1]. It is typically set as a query-retrieval task where the system retrieves a ranked list of possible covers of a given query song from a reference database. We recommend you reading [1] for more comprehhensive details of this task. 

Most of the state-of-the-art cover identification systems use pairwise comparison of two musical recordings to compute an similarity distance by using the following workflow.


- Tonal feature extraction using chroma features such as [`HPCP`](https://essentia.upf.edu/reference/std_HPCP.html).
- Post-processing of the tonal features to achieve invariance (eg. Pitch) [3].
- Computing cross-similarity between a pair of query and reference song tonal features [2], [4].
- Local sub-sequence alignment to compute the pairwise cover song similarity distance [2], [3], [6].


Recently, we have implemented the following state-of-the-art pairwise cover song identification algorithms in Essentia providing an optimised open-source implementation for both offline and realtime use-cases.

- [`CrossSimilarityMatrix`](https://essentia.upf.edu/reference/std_CrossSimilarityMatrix.html) : Compute euclidean cross-similarity between given two framewise feature arrays with an optional parameter for binarizing with a given threshold. This algorithm can be generically used for computing cross-similarity of any given input features.


- [`ChromaChrossSimilarity`](https://essentia.upf.edu/reference/std_ChromaCrossSimilarity.html) : Compute cross-similarity between a query and reference song frame-wise chroma features using cross recurrent plot [2] or OTI-based binay similarity matrix method [4].


- [`CoverSongSimilarity`](https://essentia.upf.edu/reference/std_CoverSongSimilarity.html) : Compute cover song similarity distance using various alignment constraints [2], [3] of Smith-Waterman sub-sequence alignment algorithm [6].


In the following sections, we will show how these music similarity and cover song identification algorithms can be used in both essentia standard and streaming mode workflows.

## Standard mode computation

#### Step 1

For test use-case, we select 3 songs from the [covers80](https://labrosa.ee.columbia.edu/projects/coversongs/covers80/) dataset as query song, true reference song and false reference song respectively.

- Query song

<audio src="{{ site.baseurl }}/assets/cover-song-similarity/en_vogue+Funky_Divas+09-Yesterday.mp3" controls preload></audio>  


- True reference song

<audio src="{{ site.baseurl }}/assets/cover-song-similarity/beatles+1+11-Yesterday.mp3" controls preload></audio>  


- False reference song

<audio src="{{ site.baseurl }}/assets/cover-song-similarity/aerosmith+Live_Bootleg+06-Come_Together.mp3" controls preload></audio> 


Next step is to load our query song and two reference songs using any of the audio loader algorithms in essentia (here we use `MonoLoader`). We compute the frame-wise HPCP chroma features for all of these selected songs using the utility function `essentia.pytools.spectral.hpcpgram` in the essentia python bindings.


> Note: audio files from the covers80 dataset have a sample rate of 32KHz.


```python
import essentia.standard as estd
from essentia.pytools.spectral import hpcpgram

query_filename = "../assests/en_vogue+Funky_Divas+09-Yesterday.mp3"
true_ref_filename = "../assests/beatles+1+11-Yesterday.mp3"
false_ref_filename = "../assests/aerosmith+Live_Bootleg+06-Come_Together.mp3"

# query cover song
query_audio = estd.MonoLoader(filename=query_filename, 
                              sampleRate=32000)()
# true cover
true_cover_audio = estd.MonoLoader(filename=true_ref_filename,       
                                   sampleRate=32000)()
# wrong match
false_cover_audio = estd.MonoLoader(filename=false_ref_filename, 
                                    sampleRate=32000)()

# compute frame-wise hpcp with default params
query_hpcp = hpcpgram(query_audio, 
                      sampleRate=32000)

true_cover_hpcp = hpcpgram(true_cover_audio, 
                           sampleRate=32000)

false_cover_hpcp = hpcpgram(false_cover_audio, 
                            sampleRate=32000)
```


Okay, let's plot the first 500 frames of HPCP features of the query song.

```python
%matplotlib inline
import matplotlib.pyplot as plt

plt.title("Query song HPCP")
plt.imshow(query_hpcp[:500].T, aspect='auto', origin='lower')
```

![png]({{ site.baseurl }}/assets/cover-song-similarity/query-hpcp.png)


#### Step 2


Next step is to compute the cross-similarity between the given query and reference song frame-wise HPCP features using [`ChromaChrossSimilarity`](https://essentia.upf.edu/reference/std_ChromaCrossSimilarity.html) algorithm. The algorithm provides mainly two type of methods based on [2] and [4] respectively to compute the aforementioned cross-similarity. 

In this case, we have to compute the cross-similairty between the `query-true_cover` and `query-false_cover` pairs.



Here `oti=True` parameter enable the optimal transposition index to introduce pitch invariance among the query and reference song as described in [5].

```python
cross_similarity = estd.ChromaCrossSimilarity(frameStackSize=9, 
                                        frameStackStride=1, 
                                        binarizePercentile=0.095,
                                        oti=True)


true_pair_sim_matrix = cross_similarity(query_hpcp, true_cover_hpcp)
```

Let's plot the obtained cross recurrent plot [2] between the query and true reference cover song HPCP features.

```python
plt.title('Cross recurrent plot [2]')
plt.xlabel('Yesterday accapella cover')
plt.ylabel('Yesterday - The Beatles')
plt.imshow(true_pair_crp, origin='lower')
```

![png]({{ site.baseurl }}/assets/cover-song-similarity/crp-true-pair.png)


Now, let's compute the cross-similarity between the query song and false reference cover song HPCP features.


```python
false_pair_sim_matrix = cross_similarity(query_hpcp, false_cover_hpcp)

# plot 
plt.title('Cross recurrent plot [2]')
plt.xlabel('Come together cover - Aerosmith')
plt.ylabel('Yesterday - The Beatles')
plt.imshow(false_pair_crp, origin='lower')
```


![png]({{ site.baseurl }}/assets/cover-song-similarity/crp-false-pair.png)

OR (Optional)

Alternatively, we can also use the OTI-based binary similarity method as explained in [4] to compute the cross similarity of two given chroma features by enabling the parameter `otiBinary=True`.

```python
csm_oti = estd.ChromaCrossSimilarity(frameStackSize=9, 
                                    frameStackStride=1, 
                                    binarizePercentile=0.095,
                                    oti=True,
                                    otiBinary=True)

oti_csm = csm_oti(query_hpcp, false_cover_hpcp)
```


#### Step 3

The last step is to compute an cover song similarity distance between both `query-true_cover` and `query-false_cover` pairs using local sub-sequence alignment algorithm designed for cover song identification task. [`CoverSongSimilarity`](https://essentia.upf.edu/reference/std_CoverSongSimilarity.html) algorithm in essentia provides two alignment constraints of smith-waterman algorithm [6] based on [2] and [3]. We can switch between these methods using the `alignmentType` parameter. Please refer the documentation for more details.


Let's compute the cover song similarity distance between true cover song pairs.

```python
score_matrix, distance = estd.CoverSongSimilarity(disOnset=0.5, 
                                            disExtension=0.5, 
                                            alignmentType='serra09',
                                            distanceType='asymmetric')
                                            (true_pair_crp)

# plot
plt.title('Cover song similarity distance: %s' % distance)
plt.xlabel('Yesterday accapella cover')
plt.ylabel('Yesterday - The Beatles')
plt.imshow(score_matrix, origin='lower')
```
![png]({{ site.baseurl }}/assets/cover-song-similarity/true-cover-pair-qmax.png)



Now, let's compute the cover song similarity distance between false cover song pairs.

```python
score_matrix, distance = estd.CoverSongSimilarity(disOnset=0.5, 
                                            disExtension=0.5, 
                                            alignmentType='serra09',
                                            distanceType='asymmetric')
                                            (false_pair_crp)

# plot
plt.title('Cover song similarity distance: %s' % distance)
plt.ylabel('Yesterday accapella cover')
plt.xlabel('Come together cover - Aerosmith')
plt.imshow(score_matrix, origin='lower')
```

![png]({{ site.baseurl }}/assets/cover-song-similarity/false-cover-pair-qmax.png)


Voila! We can see that the cover similarity distance is quite low for the actual cover song pairs as expected.


## Streaming mode computation

We can also compute the aforementioned similarity measures in essentia streaming mode which suits its real-time application use cases. The following code block shows an simple example of the workflow where we stream the query song HPCP feature to compute the real-time cover similarity distance between a pre-selected reference song HPCP feature.

```python
import essentia.streaming as estr
from essentia import array, run, Pool

input_stream = estr.VectorInput(query_hpcp)

# create instance of streaming cross-similarity matrix algorithm
sim_matrix = estr.ChromaCrossSimilarity(
                referenceFeature=true_cover_hpcp)

# Create an instance of the alignment algorithm 
# 'pipeDistance=True' stdout distance values for every frame
alignment = estr.CoverSongSimilarity(pipeDistance=True)

# Pool instance (python dict like object) to aggregrate the outputs  
pool = Pool()

# connect all the algorithms in a network
input_stream.data >> sim_matrix.queryFeature
sim_matrix.csm >> alignment.inputArray
alignment.scoreMatrix >> (pool, 'scoreMatrix')
alignment.distance >> (pool, 'distance')

# run the algorithm network
run(input_stream)

# Now, let's check the final cover song similarity distance value 
# computed at the end of stream.
print(pool['distance'][-1])
```




## References

[1] Joan Serrà. Identification of Versions of the Same Musical Composition by Processing Audio Descriptions. PhD thesis, Universitat Pompeu Fabra, Spain, 2011.

[2] Serra, J., Serra, X., & Andrzejak, R. G. (2009). Cross recurrence quantification for cover song identification.New Journal of Physics.

[3] Chen, N., Li, W., & Xiao, H. (2017). Fusing similarity functions for cover song identification. Multimedia Tools and Applications.

[4] Serra, Joan, et al. Chroma binary similarity and local alignment applied to cover song identification. IEEE Transactions on Audio, Speech, and Language Processing 16.6 (2008).

[5] Serra, J., Gómez, E., & Herrera, P. (2008). Transposing chroma representations to a common key, IEEE Conference on The Use of Symbols to Represent Music and Multimedia Objects.

[6] Smith-Waterman algorithm (Wikipedia, https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm).


# Background

```{contents}
:depth: 2
:local: true
```

## What is outlier detection?

Machine learning models learn to map inputs onto outputs based on patterns observed in a training set $(X_{train},Y_{train})$.  Before deploying a model, one validates its performance on unseen data $(X_{val},Y_{val})$. Now suppose there exists a measure $s(x)$ of the extent to which any unseen data point $x$ satisfies patterns observed in $X_{train}$. Additionally suppose that $s(x)$ suggests that a newly arriving data point deviates from $X_{train}$ more than any of the points in $X_{val}$. We would conclude that our model has not been validated on such a point and be suspicious of its ability to process it. This captures the essence behind outlier detection[{sup}`1`](types_of_od), which can be critical in the prevention of costly or even dangerous predictions by models operating with real world consequence.


## The anatomy of an outlier detector 

An outlier detector is simply a function $o_X(x)$ which, given an unseen data point $x$, maps it to 1 if it outlying relative to $X$ and maps it to 0 otherwise. Under the hood outlier detectors work by comparing the value of a *score function* $s_X$ to a threshold $h$, such that $o_X$ takes the form $o_X(x)=1\{s_X(x) \geq h\}$. 

Lets consider some concrete examples to lend intuition to what $s_X(x)$ might look like. The points in $X_A$ satisfy the pattern "$x_1$ is approximately equal to $x_2$", which could be captured by the score function $s_{X_A}(x)=|x_1-x_2|$. The points in $X_B$ satisfy a pattern along the lines of "$x$ is close to $(1,1)$ or very close to $(-1,-1)$", which might be captured by the score function $s_{X_B}(x)=\max(||x-(1,1)||_2, \,10||x-(0,0)||_2)$. By specifying thresholds for these score functions we divide the covariate space into inlying and outlying regions, as (not currently) shown.

```{figure} images/placeholder_1.jpg
:align: center
:alt: Placeholder
:width: 600px
A placeholder
```

In these examples the distribution underlying $X$ is simple enough that we can specify suitable score functions ourselves. But it is less clear how we would specify a rule for $X_C$, where the distribution is more complex. It is even less clear how we would do so for image, text or mixed-type tabular data. What we need are methods for *automatically* converting a dataset $X$ into a score function $s_X$. It is this conversion that defines an outlier detection method and is how different outlier detectors differ.


## Perfect outlier detectors do not exist


Before we can consider how best to automate the construction of scores function $s_X(x)$, we should consider precisely which points we would like it to assign high "outlyingness" to. If we knew in advance what outliers would look like, this would be relatively straightforward. But the nature of outliers is that they are unexpected and their form not known in advance. For the model monitoring context, given that the model is validated on unseen data from the same distribution as the training data, what we'd really like to do is to assign high scores to points that don't appear to be from this distribution, and low scores to those that do.

Given this criteria it is tempting to conclude that an objective quantification of outlyingness is captured by the probability density function $p_X(x)$. If so, the obvious way to proceed might be to obtain an estimate $\hat{p}_X(x)$ and make the score function inversely proportional, such as $s_X(x)=1-\hat{p}_X(x)$. Whilst $p_X(x)$ can be a useful object on which to focus, it is not as objective as it might seem. This is because it measures the probability *density* under a *particular representation* of the data. Changing the representation changes which points $s_X(x)=1-p_X(x)$ flags as outliers ([Le Lan & Dinh (2021)](https://arxiv.org/abs/2012.03808)). Therefore focusing on density functions merely converts the problem from that of specifying a suitable score function, to that of specifying a suitable representation.
<!-- Secondly, modelling $p_X(x)$ is hard on high dimensional domains and is particularly hard over regions of low probability density, which are precisely those of importance for outlier detection. It is often more effective to solve the simpler problem of estimating a particular decision boundary $p_X(x)<h$ to be used to determine outliers. -->

In fact, it is not possible to define outliers in an objective and universally useful manner. A point that looks outlying with respect to one representation may turn out to be perfectly inlying with respect to the patterns relevant to the task at hand. Conversely, seemingly inlying points can turn out to be outlying in consequential ways. Given this, the most robust approach is to consider a diverse **ensemble** of outlier detectors, each of which have different ideas of what an outlier looks like. The more of these detectors to which a point looks outlying, the more likely it is to be problematic to the model.


## Principled outlier detectors do exist

Although no score function $s_X(x)$ can result in the perfect outlier detector, there are still principles one can follow to design score functions that are more likely to be effective in practice. Whilst there exist many ways of categorising outlier detectors, at various granularities, perhaps the broadest is into similarity-based methods and invariance-based methods.

A **similarity-based** score function $s_X(x)$ scores $x$ based on how many "similar" points it has in $X$. Note that this encompasses methods based on probability density functions (e.g.likelihood ratios, GMM etc) as well as methods that compute distances to neighbours (e.g. KNN, LOF, OCSVM etc). The assumption underlying these methods is that if the model has seen many instances "similar" to $x$, and their associated labels, then we can trust it to process it sensibly. Similarity-based methods depend heavily one the choice of representation and/or the choice of distance/similarity metric. 

An **invariance-based** score function $s_X(x)$ identifies patterns adhered to by the vast majority of points, and scores $x$ highly if it breaks these patterns. For example if a (non-trivial) function $f$ can be identified that maps every point in $X$ to 0, then we consider inliers *invariant* to $f$ and worry when we encounter points that $f$ doesn't map to 0. We might try to identify a diverse set $f_0, ..., f_n$ of such functions and construct $s_x(x)$ accordingly (e.g. Mahalanobis, DeepSVDD). Alternatively we might learn a function $g$ that is invariably able to compress and accurately reconstruct points in $X$ and take $s_X(x)$ to be the reconstruction error (e.g. PCA, autoencoder). The assumption underlying these methods is that the model leverages patterns in the data to make decisions, so if a new point $x$ deviates significantly from patterns present during training the model might struggle to process it sensibly.

Alibi Detect offers a suite of outlier detectors, each with different underlying score functions, for use as both standalone detectors and components within overarching ensembles. Some detectors are more suitable for certain problems than others. Things to consider when choosing detectors, representations and hyperparameters are discussed in section 7. Detailed explanations of how each detector works can be found in their associated documentation. 

## The line between inlier and outlier

In many contexts concrete actions are to be taken when we encounter a point that is considered to be outlying. Where the line is drawn between inliers and outliers has tangible consequences. If the threshold is set too high then problematic outliers might be missed. But the lower the threshold is set, the more often perfectly inlying points will trigger potentially costly false alarms. 

It is important that outlier detectors allow control over this trade off. However, it is usually unclear for a given detector how to specify a value $h$ for the threshold such that a desired rate of false alarms is achieved. Instead we specify the proportion $\alpha$ of inlying data (points drawn from the training data distribution) we are willing to be falsely alerted to in order to be sensitive to outlying data. We can then estimate the corresponding threshold $h$ using the validation data, by identifying the value $\hat{h}$ exceeded by only $\alpha\%$ of instances.

For example, if we wish for a false positive rate of $\alpha=0.01$ and have 10,000 validation instances, the threshold is set equal to the 9,900th highest score. If we instead only wish to detect points more outlying than *any* of $N_{val}$ validation instances, we can set $\alpha=1/N_{val}$. We can also return estimated p-values alongside detections. If a detection is made with a p-value of 0.002, for example, this means that whilst the point could be inlying, such a high score is only obtained by the most extreme 0.2% of inlying points.

## Defining and using detectors in Alibi Detect

To demonstrate the definition and usage of detectors in Alibi Detect, we consider the particularly simple score function $s_X(x)$ that corresponds to returning the distance from $x$ to its $k$-th nearest neighbour in $X$:
```python
from alibi_detect.od import KNN

detector = KNN(k=10)
detector.fit(X_train)
score = detector.score(x)
```
We see that we first instantiate the detector with arguments relevant to the definition of the score function. In this case we choose to return the distance to the point's 10th nearest neighbour. We then produce the score function $s_X(x)$ by passing our dataset to the `.fit()` method. This function is then exposed via the `score()` method, to which new points `x` can be passed.

In order to obtain a binary prediction as to whether or not $x$ is from the training distribution, alongside an associated p-value, we must first use validation data to infer the threshold corresponding to a desired false positive rate before calling the `.predict()` method:
```python
detector = KNN(k=10)
detector.fit(X_train)
detector.infer_threshold(X_val)
pred = detector.predict(x)
```
This produces a dictionary `pred` which contains the binary prediction, p-value, score, threshold and other detector metadata. It is important that `infer_threshold()` is passed a different split of data to `.fit()`. Otherwise the threshold is inferred on data that will appear more similar to the detector than unseen data and the threshold will be mistakenly low, resulting in a false positive rate higher than that desired.

## Combining detectors into an ensemble in Alibi Detect

As discussed in section 3, we recommend ensembling predictions over diverse sets of detectors. This can be done by wrapping a set of pre-fitted detectors into an `Ensemble` object, which then handles normalisation and aggregation of scores:
```python
from alibi_detect.od import Ensemble, KNN, SVM, GMM

detector_1 = KNN(k=10).fit(X_train)
detector_2 = SVM(nu=0.25).fit(X_train)
detector_3 = GMM(n_components=5).fit(X_train)

ensemble = Ensemble(
    [detector_1, detector_2, detector_3],
    normalizer='p-val',
    aggregator='max',
)
ensemble.infer_threshold(X_val)

pred = ensemble.predict(x)
```
The `pred` dictionary then contains scores and predictions at the level of both the ensemble and the constituent detectors. Here we specified that the scores corresponding to different detectors should be standardized by using their associated p-values (we use 1-p_value to preserve positive correlation with outlyingness) and then the ensemble simply takes the max of these normalized scores as the aggregated score. So in this case instances that are considered *extremely* outlying by a single detector are deemed more outlying than those considered *quite* outlying by all detectors. 

For more details on possible normalization and aggregation strategies, see the associated methods page.


## Choosing suitable detectors, representations and hyperparameters

Whilst there are no universal rules specifying exactly which detectors, representations and hyperparameters should be chosen for a given use case, we can use knowledge of the problem domain as well as an understanding of the model being monitored to help guide our decisions.

Regarding **knowledge of the problem domain**, we might have a very high-level idea for the form outliers might take. For example, suppose we are performing outlier detection on images. It might be that we are particularly worried about encountering outliers corresponding to a malfunctioning or blocked camera. In these cases outliers would surface at the level of the raw inputs (pixels) and therefore it would make sense to apply outlier detectors directly to a pixel-based representation. Alternatively we might be particularly worried about encountering images containing previously unseen concepts, in which case it would make sense to apply outlier detectors to a semantic representation, which can be obtained by passing images through a pretrained computer vision model.

Regrading an **understanding the model being monitored**, we can consider which types of outlier the model might be particularly unlikely to handle well. The more complex our model is and the more likely it is to leverage complex interactions and fine-grained patterns, the more we will want outlier detectors to also be able to spot these. For less complex models we may wish to focus more on simple global relationships. Most detectors have a regularisation parameter that can be used to control this trade off, such as `k`, `nu` and `n_components` for the `KNN`, `SVM` and `GMM` detectors shown in the section above.

As for the representation of data to pass to detectors, usually we will want to pass whichever representation the model being monitored is trained on, here patterns it picked up on are likely to be most prominent. However for very high-dimensional data such as images or text many types of outlier will be difficult to spot in this space. We will instead want to project onto a lower dimensional space where distances between points are more meaningful. For images and text pretrained models[{sup}`2`](learnt_reps) can be useful for extracting semantic representations. For other types of data we can use more traditional dimensionality reduction techniques or use domain knowledge to explicitly specify meaningful notions of similarity between data points via a kernel which can be passed to kernel-based detectors.

If it is not clear what detector, data representation or level of regularisation to use we can simply specify multiple detectors and form an ensemble as described in the previous section.

---
## Footnotes
(types_of_od)=
:::{admonition} {sup}`1` Types of outlier detection

Sometimes "outlier detection" is used to refer to a similar but slightly different problem of filtering a dataset into inliers and outliers. Here there is just a single object of interest -- the dataset $X$ -- and all of the data points within it receive equal treatment. By contrast, the conception of outlier detection upon which Alibi Detect focuses on has two objects of interest -- a dataset $X$ and a separate data point $x$ -- and detection focuses on whether $x$ is outlying *relative to $X$*. This problem is also commonly referred to as **out-of-distribution** detection. To highlight the difference consider the example below where $X$ contains three modes, one of which contains a 2\% minority of the data. Many filtering outlier detectors would label all the data points in this minority mode as outliers. However, in the context of monitoring the inputs to machine learning models, we may not wish to identify future data points falling into this mode as outliers: the model has seen many such inputs during training and may be able to act on them perfectly well. We may instead be more interested, for example, in whether a new data point is similar to *any* of the instances in $X$. 

```{figure} images/placeholder_2.png
:align: center
:alt: Placeholder
:width: 600px
A placeholder
```
:::

(learnt_reps)=
:::{admonition} {sup}`2` Using representations learnt by the model
It might seem that the best way to detect outliers that are problematic for the model is to look for outliers in a representational space learnt by the model. Indeed we have good reason to be suspicious of points that end up outlying in these spaces. However, models only preserve information in the input that is relevant to the task at hand **on the training data**. For example a classifier of frogs against flamingos might focus only on the presence of the colours green and pink and throw away information regarding bodily features. In this case an image of a green snake might appear perfectly inlying with respect to the model's learned representation, whereas we would like it to be flagged as an outlier.
:::

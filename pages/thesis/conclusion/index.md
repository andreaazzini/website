---
title: Conclusion
---

In this thesis, we have created a pipeline which is able to precisely
retrieve quantitative information of food, such as volume and weight, by
using only deep learning techniques. We achieved this goal by devising a
particular combination of segmentation, 3D shape reconstruction and
further script manipulations on the results obtained from the networks.
We also faced the challenging problem of synthesizing a proper training
set with a novel approach, i.e. creating 2D synthetic views of food from
3D models. This way we discovered interesting results, generalizable to
a wide variety of related problems in computer vision.

The first main contribution of this work, i.e. the pipeline, is able to
output the absolute volume of the ingredients in a picture. The pipeline
first component is the segmentation network, which is used for 3 main
reasons:

-   refining the 2D pictures to help the volume reconstructor network

-   finding the ratio between a common object of standard sizes (i.e. a
    Stabilo highlighter) used as a reference and the objects of unknown
    sizes to be quantified

-   creating an additional 2D view from each input picture, which is the
    same picture without the reference object. This picture is used
    later in the computation of the volume by subtraction

We surveyed different segmentation techniques and chose SegNet
[[5](references#badrinarayanan2015segnet)] for its high performance on difficult test
sets. We provide our implementation in TensorFlow of encoder-decoder
architectures based on the SegNet work and enriched with ideas such as
strided convolutions. Our code has already been used by other
researchers on different training sets and has received enthusiastic
reviews.

As the 3D reconstructor we chose 3D-R2N2 [[11](references#choy20163d)]. It is used for:

-   recreating the precise 3D shape of the objects in the 2D pictures,
    given as input, by keeping the right proportions between them

-   reconstructing two 3D scenes, one with a reference object of
    standard sizes (i.e. Stabilo highlighter) and one without it, and
    obtain the measure in voxels of the reference object by subtraction
    of voxels from the two scenes. The reference object has standard and
    known sizes, therefore all the other measures can be then obtained
    by proportion

After we designed the pipeline structure, to make its component networks
work efficiently and together, we had to fine-tune them, as well as
devise proper training sets and experiments.

We have built a fully synthetic dataset mainly for 2 reasons:

-   **Scalability in our context**: indeed, we made the pipeline work
    with five classes of ingredients, but the overall training process
    we devised is easily scalable with many more

-   **Generalization**: one of the main challenge in data driven
    computer vision, and in general in machine learning, is to find a
    proper amount of labeled data. Our study on building a proper
    synthetic training set can help building new training sets in many
    contexts

Regarding the generalization point just mentioned, here are some of the
training set features that we studied:

-   **Single- against multiple-object pictures**: pictures with just one
    object at a time are compared with pictures with multiple instances
    of objects

-   **Texture interpolation**: we started from approximately 20 textures
    from each ingredient, and we studied the effect of combining them to
    obtain new textures

-   **Background complexity**: we scraped pictures with colors similar
    to the ingredients represented, in order to force the network to
    learn features non-color related

-   **Variation in lighting and shadow**: we study the effects of lights
    from different angles and with different colors and intensities. We
    also set different kind of shadow features.

By adding these features we got increasingly better results. However,
only when augmenting the variation in lighting and shadow, we got a
network able to not only distinguish the objects from the background,
but also to make precise predictions on each ingredient class.

This finding is extremely important. First, it provides a major reason
to use synthetic 3D models to use their 2D views in training sets.
Indeed, by using 3D models it is relatively easy to produce thousands of
variations in lighting and and camera orientation out from very few
models. This also gives directions on what are the most important
features to focus on when building such synthetic training sets.

The component networks we used in the pipeline are context-agnostic,
therefore all the intermediate findings from our experiments can be used
in other research contexts. One finding, is the capability of the 3D
recurrent reconstructor network to keep the proportions between multiple
objects in the same reconstruction, that we proved.

There are many directions, for possible future works:

-   **Multitasking network**: the first one, better described in Chapter
    4, is to leverage the fully convolutional
    nature of our pipeline, to devise a single network, for instance by
    using the multitasking principle, and train it an an end-to-end
    fashion. This can provide insights not only about the quantification
    problem, but also about the behavior of convolutional neural
    networks when applied to computer vision

-   **GANs**: utilize Generative Adversarial Networks when creating the
    synthetic training set, to provide an automatized step in the
    synthesis process

-   **Pipeline extensions**: extend our pipeline with further blocks, to
    add more features or fine-tune the existing ones

-   **Related problems**: integrate the quantification pipeline with
    other systems, to solve higher level tasks, for instance medical
    ones

Quantification has a wide range of practical applications. Furthermore,
it opens the doors to many other complex tasks in Artificial
Intelligence. Therefore, we think it is worthwhile to push forward the
results of this thesis, and extend them in future studies, as well as
implement them in real-world applications.

---
title: Introduction
---

### Context

Deep learning has become a popular research subject, thanks to its achievements and results on many subfields and real-life problems, such as computer vision [[28](references#krizhevsky:2012)][[34](references#lin2013network)][[45](references#simonyan2014very)][[50](references#szegedy2015going)][[37](references#long2015fully)][[20](references#he2016deep)], natural language processing [[48](references#sutskever2014sequence)][[10](references#cho2014learning)][[27](references#kalchbrenner2014convolutional)][[21](references#hermann2015teaching)][[29](references#lample2016neural)], robotics [[31](references#lenz2015deep)][[38](references#mnih2015human)][[32](references#levine2016learning)], cancer detection [[15](references#esteva2017dermatologist)][[19](references#havaei2017brain)], autonomous cars [[24](references#huval2015empirical)][[8](references#chen2015deepdriving)], and more.

Despite its success in specific domains, deep learning is generally
advancing at small incremental steps, and its methodologies have been
applied to research topics that share a lot of implementation and
practical aspects, but still are very domain-specific. For instance,
convolutional neural networks have been tested thoroughly in a lot of
fields, and have been pushed beyond the state-of-the-art models of the
past. New models have been proposed to solve very peculiar problems by
changing a few details only, such as the nature of the loss function to
be minimized, or the number and type of layers to be included in the
network. Techniques such as Dropout [[23](references#dropout)] and Batch Normalization
[[25](references#ioffe:2015)], which will be described in the following chapters, have
now become a standard in the design and implementation of deep neural
networks, because they have been proved to improve the performance and
the results of the models dramatically.

It may seem that deep learning is a subject which has been already
studied, and cannot offer more compelling research questions. However,
only recently, new methods such as Generative Adversarial Networks
[[18](references#GANs)] have revolutionized the way networks learn by competing against
each other. From a microscopic viewpoint, they only use the building
blocks of convolutional or recurrent neural networks, but it is the way
those building blocks are put together to create an innovative pipeline
that made those models incredibly successful. Similarly, with our
thesis, we do not aim at creating new models to solve already existing
problems, and push the results beyond state-of-the-art. Instead, our
objective is to create a pipeline of independent modules, aimed at
solving a difficult innovative problem, which we call *Quantification*.

### Problem Statement

Human visual system is a masterpiece of nature. A large contribution to
our skills and intelligence is due to vision. For us, it is enough to
take a quick look at a picture, to being able to answer a lot of
questions about it. We are able to reproduce an internal representation
of the 3D models on the scene, and by using that, plus our prior
knowledge of the objects represented, we can estimate detailed measures.
For instance, by looking at a picture of a man holding a very small cat,
we can naturally estimate a narrow range for the animal’s weight.

Computers are still far from being able to solve as efficiently many
tasks that are natural for us. Nonetheless, improvements in computer
vision are making machines able to solve increasingly harder questions.
Before discussing the main question we want to answer in this thesis,
about quantification, and what kind of quantitative information we want
to retrieve in pictures, we first review related problems:

-   **Classification [[43](references#russakovsky2015imagenet)]**: Classification task
    is about answering the question “what is in the picture?” with a
    single label, i.e. banana. It doesn’t matter how many objects are in
    the scene, the answer is the label of the object detected with
    higher probability. There is no quantitative information in this
    kind of predictions.

-   **Object detection [[16](references#everingham2010pascal)]**: Object detection task
    is more advanced, and targets the question: “where are each object
    in the scene?”. Thus, it is about assigning a frame to each object
    in the scene. By counting the detected frames, it is possible to
    have a first quantitative information, the number of objects in the
    scene, with their corresponding classes.

-   **Segmentation [[16](references#everingham2010pascal)]**: Semantic segmentation, or
    pixelwise classification, is about assigning a class to each pixel
    in the scene. It is more precise than object detection in terms of
    the estimated area, and gives another quantitative information in
    term of the pixels’ number.

These tasks are not directly related to the 3D shape of objects, which
is in fact really important for a broader understanding. Tasks regarding
3D information are:

-   **Object orientation [[36](references#liu2016upright)]**: A first example of a
    3D-related task is object orientation, where the goal is to
    determine the orientation of objects relative to some coordinate
    system.

-   **Depth estimation [[35](references#liu2015deep)]**: Depth estimation, is about
    producing a depth map, to represent how much the objects in the
    scene were close to the camera.

-   **3D reconstruction [[47](references#su2015multi)]**: Finally, an interesting task
    is 3D reconstruction, which is about recreating the 3D shape of
    objects from one or multiple 2D pictures representing the object
    taken from different camera angles.

The main goal of this thesis is to develop a method to answer the
question “how much of each object is in a scene?”. By “scene” we mean
the representation of a group of objects, via one or multiple pictures
of the same group, taken from different angles. By “how much”, we refer
to something more detailed than just counting the number of objects in
the scene. In fact, we want to retrieve as many quantitative information
as possible. In the domain we chose to focus, food ingredients, these
are for instance the calories of each ingredient, how many person can
one serve with that food, even suggestions about possible recipes. In
particular, we are interested in retrieving weight, since a lot of other
information can be obtained from it.

Our objective is not only to devise an abstract method to answer this
question, but also to solve it practically and provide a proof of
concept that works with real ingredients.

### Proposed Solution

Our main objective to verify if it is possible to quantify objects using
only deep learning techniques. In order to test this, we have performed
a lot of experiments to find the most suitable models that could be able
to achieve our goals. We decided to divide the problem into two main
parts, that are *segmentation* and *3D model reconstruction*. Our
pipeline has then been built based on these two concepts.

The segmentation module of the pipeline is responsible for defining the
shape and contour of the objects. As we will describe in the following
sections, the segmentation module operates using a fully-convolutional
autoencoder, that is fed with images representing objects belonging to
different classes, encodes the object features and decodes them while
trying to label every pixel according to their already known class.

The reconstruction module needs the results of the segmentation module
to reconstruct the 3D shape of the original object. This process paves
the way for the final quantification. If the two modules that constitute
the pipeline are conceptually very simple, there are however a lot of
unresolved issues. Firstly and most importantly, the training set to be
used needs to include not only the input images and their labels, but
also a 3D model as label for the reconstructor. Secondly, the problem
needs to be contextualized, because quantifying an object may mean very
different things. Finally, there is the problem of retrieving not only
relative measurements but absolute ones, so there is the needing to
devise a precise criteria to estimate the reconstructed values. Before
diving into the implementation specifics, we identified the best ways to
solve these issues.

#### Synthetic Data and Generalization

To address the problem of needing a very rich dataset, we concluded that
the search for a preexisting dataset on the internet was practically
infeasible. In fact, it is difficult, if not impossible, to find a
dataset which satisfies all the requirements we have listed above.
However, it is certainly possible to synthesize the dataset on our own,
provided that it is done considering all the variation and complexity of
the data we need to reproduce.

Although generating the data seemed a reasonable approach, synthetic
images can be very different from their real counterparts, thus making
the results extremely biased and inaccurate for a real-case scenario.
The way we decided to tackle this problem is not only to generate data
trying to make our synthetic images as close to their real twins as
possible, but also to study the *generalization* power of our pipeline.
With *generalization*, we mean the potential of changing the test set
from a synthetic to a real one, without losing prediction capabilities.

More specifically, during our experiments, we have trained our models
using only synthetic images. Then, module by module, we have tested the
model against real images that we have collected. Obviously, the real
images represent the same kind of objects we have trained the network
with, only they are real pictures, not generated by a computer. If
proved successful, generalization is a very powerful technique because
it will demonstrate that, within certain contexts, it is possible to
synthesize data, without being worried about losing prediction power
over the real data with a certain degree of confidence.

![Synthetic (i.e. computer generated)
images.](/assets/images/final_training_set.png)

#### Contextualization

We chose food as a good environment to generate the data for the focus
of this work. Food has interesting challenges, like different textures,
colors, and shapes, that make it perfect for the kind of research we
want to conduct. It is also relatively easy to find real images of food
to test generalization, and it is also relatively easy to take pictures
of our own. Finally, food quantification is directly related to
nutrition and diet. Being able to know, for instance, how much of a
particular food there is in a picture is directly related to the
nutritional values of the food itself, thus making the problem
interesting also from an application perspective. If our models work, it
would be relatively easy to create an application that, given the images
of that food, it is able to predict its nutritional values instantly.
The research question is then not only interesting from an architectural
perspective, or because its generalization capabilities, but also
because of its applicative nature. Of course, we have chosen food for
the aforementioned reasons, but there could be way more context that
will make our models worth using.

### Structure of the Thesis

The subject we have worked on is complicated, and needs to be addressed
before diving into implementative aspects of our research. For this
reason, we have organized the thesis in a way that makes it
understandable without jumping from chapter to chapter. The structure of
the thesis is the following.

-   **Background**: in the Background chapter, we talk about the
    theoretical foundation that constitutes the basics of neural
    networks and Deep Learning, which are necessary to fully understand
    the thesis. In this chapter, we will first point out the main
    historical events that made Deep Learning so important for modern
    research. Then, we will cover the architectural basics of neural
    networks, including not only their structure, but also explaining
    how they are trained successfully. The training process issues are
    in fact fundamental in order to understand why different, more
    advanced and deeper models have been introduced (e.g. convolutional
    neural networks). We will finally explain how these advanced models
    work, and why are important for our research.

-   **Materials and Methods**: in this chapter, we elaborate the
    terminology and objectives of our thesis. They are fundamental
    before introducing our dataset and pipeline. About the dataset, we
    will explain how it has been generated, providing every detail that
    has been considered to make it as accurate as possible. About the
    pipeline, it is necessary to explain in detail how it is composed,
    how the modules are structured and the way they contribute in order
    to achieve the research goal.

-   **Experiments**: in the Experiments chapter, we list and show the
    results of all the experiments we have performed in order to verify
    thepipeline capabilities. The chapter itself is divided into two
    main section, one for each of the main modules that constitute the
    quantification pipeline, which are *segmentation* and
    *reconstruction*.

-   **Conclusion**: we conclude our thesis considering the effectiveness
    of our methods, considering all the issues we have encountered
    during the experiments. We also list a series of possible future
    improvements that could be investigated further.

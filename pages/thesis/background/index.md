---
title: Background
---

Deep learning is a machine learning branch that has recently gained a
lot of attention because of the breakthroughs it accomplished in fields
such as computer vision, speech recognition, language translation and
many more. Similarly to the majority of machine learning branches, its
aim is to build models that can, through a series of subsequent training
iterations over the input data, learn to make accurate predictions on
some predefined context. The way deep learning differs from other
machine learning methods lies in the nature of the models it builds in
order to achieve learning: deep neural networks.

This background chapter is a summary of the theoretical information the
reader needs in order to understand the innovative aspects of our
thesis. During our experience as guest students at the Institute for
Applied Computational Science at Harvard University, we had the chance
to intensively study the subject and to come up with most of the ideas
that will be constituting the core of the thesis itself. The same
notions have also been explained in the context of our two deep learning
workshops during the ComputeFest 2017 event at Harvard.
[[4](references#azziniconserva2017)] Organized by the Institute for Applied
Computational Science, ComputeFest is an annual two week program of
knowledge- and skill-building activities in computational science and
engineering. Similarly to the objectives of this chapter, our workshops
aimed at providing the students with the theoretical and coding
foundation on the subject of deep learning.

We first introduce neural networks historically and architecturally.
Later on, we are analyzing the peculiarities of deep neural networks
during training, i.e. particular characteristics and issues of the
learning process that are specific in the context of neural networks.
Finally, more advanced architectures, which have achieved
state-of-the-art results in many different tasks, will be introduced.
These architectures are at the core of the technology that has been
built for the purposes of this thesis, and are thus fundamental to fully
understand the next chapters. Deep learning is an extremely empirical
and abstract research field. As a consequence, it very important to
focus on the basic concepts that constitute the building blocks of
artificial neural networks.

### History of Neural Networks

Let’s dive into a first historical background of neural networks. It is
important to understand why certain bricks have been used to build what
we know as modern neural networks. We will proceed in chronological
order, because it also emphasizes why certain historical conditions or
technologies made the difference for the full development of neural
networks as a model and deep learning as a subfield of machine learning.

#### The Early Age of Neural Networks

The first supervised deep feedforward architecture [[26](references#ivakhnenko:1966)]
was far from being a modern neural network, which we are describing in
the following sections, but it is important to point out, because it
shares some peculiarities with feedforward neural networks. However, the
characteristic that has contributed the most to the full development of
practically trainable neural networks is backpropagation, which was
first introduced by Werbos in 1975. [[52](references#werbos:1975)]

Backpropagation gave us huge hints on how to train neural networks
faster. It normally couples with algorithms such as Gradient Descent to
achieve its objective of minimizing a loss function, as we will see
later on in this chapter. During the 80s, a lot of research efforts have
been made on a field which is strongly related to parallel distributed
computing: Connectionism. Connectionism proposed, for the first time,
the idea of an interconnected model comprised of simple units.
[[42](references#rumelhart:1986)]

For a long time, neural networks did not really make any advancements.
There are several reasons why that was the case. First, neural networks
are computationally expensive models to train. In problems like
regression or classification, there were much easier machine learning
models that were easier to deal with (e.g. Random Forests or Support
Vector Machines). Second, neural networks typically require very big
input data, especially during training. There were not huge training
sets when the first theoretical discoveries emerged, so it was pretty
difficult to determine the power of the neural models. Third, even with
modern CPUs, the deeper the network, the slower the learning process. In
the last decade, there has been a lot of research and development
focusing on GPUs and parallel computational devices.

This effort led to very fast improvements in subfields like Computer
Vision, where the neural networks involved (i.e. mainly Convolutional
Neural Networks) can leverage this fast parallel processing to
drastically improve their training accuracies.

#### The Breakthrough of Deep Learning

Deep learning is a subject that has brought artificial neural networks
on another level. In particular, the idea behind the term deep learning
is very simple. The networks that model the problem on which we want to
make inference and prediction becomes very deep in terms of number of
layers that constitute them. The model is supposed, then, to be able to
recognize an incredibly large number of independent features and
patterns within a specific learning task. As expected, however, big
models require an enormous amount of data. That is why deep learning is
often associated with another hyped modern field of Computer Science:
Big Data.

The first time the term deep learning has been introduced can be traced
back to the year 2000. [[2](references#aizenberg:2000)] The early years of the new
millennium gave birth to a high number of papers related to topics like
convolutional and recurrent neural networks, which we will discuss in
the following sections. However, if we try to understand when Deep
Learning has really become a famous subject for both researchers and the
industry, we discover that the early 2010s were actually its clamorous
years.

![trend](/assets/images/trend.png)

The figure above shows the normalized search trend of
the deep learning topic on Google. It is pretty evident the resemblance
with an exponentially growing function. However, the turning point of
the curve is 2012, when several events showed the true potential of deep
learning not only in theory, but also in practice.

First, in 2012 Google themselves showed how a neural network could be
trained to recognize high-level objects from unlabeled images.
[[39](references#ng:2012)] Furthermore, deep learning was started to be used in the
context of competitions such as the ISBI Segmentation of neuronal
structures in EM challenge and, more importantly, the ImageNet Large
Scale Visual Recognition Challenge (ILSVRC). The network which was used
by Alex Krizhevsky et al. at ILSVRC in 2012 [[28](references#krizhevsky:2012)] was
fundamental to show the incredible potential of deep convolutional
neural networks to solve problems related to image recognition, such as
classification, object detection and so on. Thanks to the paper by
Krizhevsky et al., ILSVRC became one of the most famous competitions for
deep learning.

In particular, convolutional neural networks became a very interesting
research topic, due to its surprising results on many applications. As a
matter of facts, two papers coming out from ILSVRC participants in 2014
investigated even more into the behavior of deeper and deeper networks:
VGG [[46](references#simonyan:2014)] and GoogLeNet [[49](references#szegedy:2014)]. These new studies
really helped to give insights on how bigger models should be built to
be both trainable and performing.

### Architectural Basics of Neural Networks

Neural networks are computational models that, similarly to their
biological counterparts, are made of a large number of units (artificial
neurons) that are first arranged into layers, and then linked together
through a series of interconnections between these layers. There are
several ways which these connections can be wired in.

The simplest example of neural interconnection occurs when no cycle is
formed by all the connections within the network. Neural networks that
belong to this special class of networks, built with this simple and
straightforward rule, are called feedforward neural networks. In this
kind of networks, information flows from one direction to another in an
unambiguous fashion, from input to output, through hidden layers.

The more the number of layers which the neurons are organized in, the
deeper the network is said to be. Deep learning, from a mere conceptual
perspective, can be itself seen as a machine learning branch, in which
the models that are studied are deep neural networks. These models are
inspired by the way biological neurons are connected by axons in human
brain. However, what is referred as neuron can be called unit,
especially when the biological metaphor fades (it will be more evident
when RNNs and CNNs are introduced). Feedforward neural networks are
considered very adherent to the biological metaphor, not only because of
the nature of their connections, but also the way neurons fire.

The conditions which determine if an artificial neuron fires depends on
another design choice: the so-called activation functions. In principle,
any function which is able to define what it means for a neuron to be
fired, given certain preconditions, can be regarded as an activation
function. However, due to how the training process works in the context
of neural networks, all these functions need to retain an important
property: differentiability. That is the reason why any kind of step
function, i.e. functions that abruptly change their value, albeit being
piecewise constant for every input, cannot be used as activations, due
to their non-differentiability. The reason why this property is crucial
for neural activations relies in the way weights and biases are updated
during the training process that will be shown in the following
sections. This process, called backpropagation, leverages a gradient
descent based optimization. The gradients that are calculated at
training time make the differentiability of the activation functions an
essential requirement.

Even if the concepts presented so far are not very abstract and
complicated, the way they translate into mathematics, and consequently
into code, might be fuzzier. Each connection between the neurons that
compose the network can be represented numerically and graphically by
numerical values (weights) and arrows connecting two neurons
respectively. Additionally, each neuron is characterized by its own
overall bias, a numerical value that affects each connection towards
that specific neuron. Weights and biases typically need to be
initialized before the model is actually trained. The way this
initialization is performed is by sampling the values from some
probability distribution (e.g. a truncated normal distribution, or even
a uniform distribution). Each initialization affects the training
process, as will be clearer in the following sections. Therefore,
investing time to find a good initialization for a particular learning
context might be a clever approach.

As it was mentioned above, feedforward neural networks, which are
regarded as the simplest subset of artificial neural models, are
characterized by their simple monodirectional connections. More
specifically, these connections between layers of neurons are defined
such that they cannot form any cycle. This property is important,
because it defines how the firing of each neuron at each layer of the
network depends only on the previous layer activations. Feedforward
networks are in fact inherently time-independent, meaning that the
output of their outputs depend only on the nature of the input, weights
and biases of the network. Feeding a feedforward neural network with the
same input at different timesteps will not change the output of the
network itself. However, there are still several ways in which neurons
can be connected in order to produce this time-independent output. As it
turns out, this neural interconnection is fundamental in determining how
well (or poorly) the model will be able to learn.

![Neural Network](/assets/images/neural-network.png)

The simplest way to connect the neurons from one layer to another is by
means of so-called fully connected layers. Neurons in fully connected
layers have connections to every single activation from the previous
layer. The way the input neurons are mapped to a hidden (or output)
neuron can be seen mathematically as the activation function of that
layer, applied to the weighted sum of the input data, plus the overall
bias of the hidden neuron.

$$a\_i = \\sigma(\\displaystyle\\sum\_{i=1}^n\\sum\_{j=1}^{m} w\_{i,j}x\_i + b\_i)$$

#### Training

Before studying how to train a neural network, it is important to revise
how training and evaluating models are approached overall as problems in
machine learning. Training and evaluating models require some knowledge
of optimization theory, and they are inherently interesting and
challenging. As a matter of fact, only after training a model and
evaluating its inference capabilities, it is possible to come to certain
conclusions on its goodness. In order to analyze a topic like this one,
it is necessary to make some assumptions on the nature of the dataset.

![Neural Network Schema](/assets/images/neural-network-2.png)

The dataset considered in this context is assumed to organize and
structure information in couples in the form $(x\_i, \\hat{y}\_i)$,
representing the input data and the labels. The input data is fed to the
model, which can be thought of as a black box that produces an output
given an input. A label $\\hat{y}\_i$ represents how the output $y\_i$ of
the model should look like, given the input $x\_i$. A couple of the form
$(x\_i, \\hat{y}\_i)$ is called labeled input.

When a complete dataset of labeled data is available, the problem of
make inference can be reduced to a supervised learning task. A notorious
and standard way to train and subsequently validate a supervised
learning model takes advantage of a first split of the dataset into two
main parts: the training and test sets. These sets are meant to have
different purposes: while the training set is used to train the model,
the test set is used to assess the accuracy of the model predictions.
This way, the accuracy test will immediately spot issues like
overfitting, i.e. situations in which the trained model learns to
recognize the training data only, without generalizing to the entire
dataset.

The equation and figure above show how each activation, being it
both the output of one layer and the input to its next one
($x\_i = a\_{i-1}$), increasingly depends on previous layers’ activations.

#### Backpropagation

There are some similarities between human brain and artificial neural
networks, like the concepts of firing and interconnection between
different neurons. However there are also differences, as well as open
questions. A big challenge still unsolved is understanding how human
brain learns. According to the “single algorithm hypothesis”
[[14](references#dean2012three)] , there is only one learning algorithm behind the
different structures of the brain. Experiments that suggested this
hypothesis showed that neurons used in visual tasks can serve in the
auditory system, if trained properly. If this hypothesis is true, even
at some extent, to find “the” learning algorithm would mean find the
*Holy Grail* of learning.

Like human brain, also artificial neural networks are mainly successful
because of their capability to learn from data. Hilton
[[22](references#hinton2007recognize)] surveys some different strategies that can be
used to design a learning mechanism. Among them, Backpropagation is one
of the most famous and successful one designed so far.

Backpropagation is usually implemented in conjunction with Gradient
Descent. Gradient Descent is a heuristic to find the minimum of the cost
function with respect to the parameters in a model. As described before,
the cost function is a measure of how well a model is performing, thus
to find its minimum with respect to the model’s parameters means to find
the optimal configuration of that model to solve the precise problem
described by the cost function itself. A model is ultimately a function,
so in principle the problem of finding the global minimum of the cost
function can be solved deterministically.

However, in case of models with millions of parameters, like neural
networks, an analytical solution is computationally prohibitive. This is
why an heuristic like Gradient Descent is so useful. Let us first see
how it works in case of a linear regression problem. Suppose we have a
set of points as below

![Linear Regression](/assets/images/regr.png)

that we want to fit with a line equation such $y = mx + b$, where m is
the line’s slope and b is the line’s y-intercept. To find the best line
for our data, we need to find the best set of slope m and y-intercept b
values.

To be able to apply gradient descent, we first define a cost function,
that in this case can be the sum of the error over all the points.

$$Error\_{(m,b)} = \\frac{1}{N} \\sum\_{i=1}^N {(y\_i - (mx\_i + b))^2}$$

It is a common practice to
take the sum of the squared error, to have only positive values.

The algorithm starts with randomly initializing the parameters. Then, it
computes the derivative of the function in that configuration, with
respect to all the parameters. This derivative express the direction in
which the cost function decreases with the highest slope. Then, the
product of the parameters and a value called learning rate is added to
the parameters themselves. This change causes a decreasing of the cost,
i.e. better predictions. This upgrading steps are repeated until the
cost function reaches a local minimum. In fact, there are not decreasing
directions close to a local minimum.

We saw how gradient descent works in case of a simple linear model. With
Neural Network, the idea is the same, i.e. finding the derivative of the
cost function with respect to the parameters, that in this case are the
biases and the weights.

$$\\dfrac{\\partial C\_{X}}{\\partial w} , \\dfrac{\\partial C\_{X}}{\\partial b}$$

However, there is an additional
complexity: to find these derivatives is not straightforward, since each
neuron is connected to each other. Luckily, there is a way to find these
derivatives, through 4 equations, called the equations of
Backpropagation.

$$\\delta^L = \\nabla\_a C \\odot \\sigma'(z^L) \\tag{BP1}$$

$$\\delta^l = ((w^{l+1})^T \\delta^{l+1}) \\odot \\sigma'(z^l) \\tag{BP2}$$

$$\\frac{\\partial C}{\\partial b^l\_j} =\\delta^l_j  \\tag{BP3}$$

$$\\frac{\\partial C}{\\partial w^l\_{jk}} = a^{l-1}\_k\\delta^l\_j\\tag{BP4}$$

One main idea behind Backpropagation is to express the needed
derivatives as a function of a quantity called “error”. The error
$\\delta^l\_j$ of neuron $j$ in layer $l$ is:

$$\\delta^l\_j\\equiv\\frac{\\partial C}{\\partial z^l\_j}$$

This is useful because there is a straightforward way to compute the
error with all the information already available in a forward pass, i.e.
as a function of the input and the neural network weights and biases.
After the error is computed, then the needed derivatives are already
expressed as a function of the error itself. We provide at
[[4](references#azziniconserva2017)] a visual sequence we created to show how the
forward and backward step of Backpropagation works.

The algorithm works this way. The input data is provided to the network,
in form of a single element or a batch of elements from the training
set. Then, using the forwarding equation, i.e.
$z^{l} = w^l a^{l-1}+b^l$, the activation function is computed each
layer after another. Eventually, the activation function at the final
layer is computed as a function of the activation of its previous layer.
Here, BP1 is used to compute the error in the last layer as a function
of the activation in the last layer. This is the starting point for the
backward pass: in fact, the error in the penultimate layer can be
computed through the BP2 as a function of the error in the last layer.
In the same way, the error in each layer can be obtained from the value
of the subsequent one. Eventually, the error at every layer is
available, as well as the needed derivatives thanks to BP3 and BP4.

At this point, the network’s parameters can be updated, via the
derivatives and the learning rate.

$$w^l \\leftarrow w^l-\\frac{\\eta}{m}\\sum\_x\\delta^{x,l} (a^{x,l-1})^T$$

$$b^l \\leftarrow b^l-\\frac{\\eta}{m}\\sum\_x\\delta^{x,l}$$

Each step of Backpropagation and updating is called an epoch. The number
of epochs needed to train a network depends on many factors, such as the
complexity of the problem, the number of inputs used at each epoch (it
can varies from 1 to all the elements in training set), the learning
rate and so on.

### Architectural Issues of Neural Networks

The framework that has been introduced in the previous section provides
precise indications on how to build a neural network from square one.
Those instructions are not only valid theoretically, but they also hold
from a practical viewpoint.

However, there are several problems that can be spotted, both in theory
and in practice. As a matter of fact, we can prove that, the deeper the
network, the more difficult the learning process becomes. Also,
overfitting has been mentioned above, but it has not been addressed
concretely.

The solutions that have historically introduced have been also used also
to build our models, thus being interesting not only for their general
value, but also for our particular research problem.

#### The Vanishing Gradient Problem

As we have already mentioned in the previous sections, adding layers to
a deep neural network does not automatically translate in a
substantially better model accuracy. Counterintuitively, it can happen
that the accuracy drops even if the number of parameters is increased.
In fact, our intuition suggests that, by adding complexity to the
models, they should be able to identify more abstract and composite
features, thus improving their learning tasks.

What happens from a microscopical standpoint is that gradients tend to
get smaller as we move backward through the hidden layers of our neural
network. This problem, which is one of the most acknowledged problems in
at least vanilla neural networks, is called the *vanishing gradient
problem*. It also has a symmetric and equally worrying issue, called
*exploding gradient problem*, which does the opposite, that is making
the gradients higher and higher in the first layers of the network. In
both problems, we can say that backpropagation makes gradients
unstable.

To address the vanishing gradient, we can first translate its conceptual
description into an equation. Consider a very simple network with three
hidden layers made of a single neuron. Then, backpropagating the
gradients towards the first hidden layer translates into:

$$
\\frac{
\\partial C}{
\\partial b\_1} =
\\sigma^{'}(z\_1) w\_2
\\sigma^{'}(z\_2) w\_3
\\sigma^{'}(z\_3) w\_4
\\sigma^{'}(z\_4)
\\frac{
\\partial C}{
\\partial a\_4}$$

It is now clear that the gradients at the first layer (equation above)
are extremely influenced by the nature of the
derivative of the activation function $\\sigma$. For instance, the
sigmoid derivative has a maximum value of $0.25$, making the result of
the multiplication above smaller the deeper the network.

One way to address the problem is by means of new activation functions
that do not present the problem of the sigmoid function. A notorious
example is the Rectified Linear Unit function (ReLU). [[17](references#glorot2011deep)]

$$ReLU(x) = max(0, x)$$

For a logistic function, since the gradient can become arbitrarily
small, we can get a numerically vanishing gradient by composing several
negligible logistics, problem that gets worse for deeper architectures.
For the ReLU, as the gradient is piecewise constant, a vanishing
composite gradient can only occur if there is a component that is
actually $0$, thus introducing sparsity and reducing vanishing gradients
overall.

#### Regularization and Dropout

Vanishing gradient is one of the most relevant and notorious problems in
deep learning. Unfortunately, there are some other problems that are
inherently (and negatively) affecting the training process. One of the
most common problems in machine learning is overfitting. It occurs every
time training data are fit by an overly complex model. The model itself
is not really looking at the nature of the problem it is trying to
learn, but rather on the shape of the input data. The issue is exposed
when the model is finally evaluated with a different set of test data,
revealing how the model cannot understand the underlying relationship
between input point.

There is a very long set of proposed workarounds to deal with
overfitting. Most of these methods are common within the machine
learning context, and are not interesting only neural networks. An
example is regularization. When facing overfitted settings, one
noticeable thing is the fact that weights and biases tend to be way
greater than their non-overfitted counterparts. Weight decay, or L2
regularization, works by adding a regularization term to the cost
function, in order to deal with this weight explosion.

$$C = -\\frac{1}{n}\\sum\_{xy}[y\_j\\ln{a\_j^L} + (1 - y\_j)\\ln{(1 - a\_j^L)} +\\frac{\\lambda}{2n}\\sum\_w w^2]$$

Equation uses a $\\lambda > 0$ as regularization parameter, that makes
the network prefer to learn smaller weights, rather than larger.
Ideally, large weights are considered only when they considerably
improve the first part of the cost function, or at least better than
their square. $\\lambda$ is a hyper-parameter which needs to be tuned to
either push for the small weights preference or the original cost
function, when $\\lambda$ is large or small, respectively. In the context
of neural network, we need to consider how the gradients are calculated
after introducing the regularization.

$$\\frac{\\partial C}{\\partial w} =\\frac{\\partial C\_o}{\\partial w} +\\frac{\\lambda}{n}w$$

$$\\frac{\\partial C}{\\partial b} =\\frac{\\partial C\_o}{\\partial b}$$

$$b\\leftarrow b -\\eta\\frac{\\partial C\_o}{\\partial b}$$

$$w\\leftarrow w -\\eta\\frac{\\partial C\_o}{\\partial w} -\\eta\\frac{\\partial\\lambda}{\\partial n}w = (1 -\\frac{\\eta\\lambda}{n}) w -\\eta\\frac{\\partial C\_o}{\\partial w}$$

A smoother version of L2 regularization is L1 regularization, where the
weights get shrinked much less than with L2.

$$C = C\_o +\\frac{\\lambda}{n}\\sum\_w |w|$$

$$\\frac{\\partial C}{\\partial w} =\\frac{\\partial C\_o}{\\partial w} +\\frac{\\lambda}{n} sgn(w)$$

L1 and L2 regularization are good workarounds to deal with overfitting,
but not only they are not proven to be always effective, but also
introduce another hyper-parameter to be tuned. The most important
characteristic of these regularization methods, however, is the fact
that they do not try to inspect the model, to act on the exact way they
tend to overfit. Instead, they act on the cost function to somehow
adjust the results. A much more powerful method to reduce the effects of
overfitting neural networks is a method called Dropout.[[23](references#dropout)]

In an ordinary setting, we would train a network by forward-propagating
the input through the entire network, and then backpropagating to
determine the contribution to the gradient. With Dropout, we take a
slightly different approach, that is we randomly and temporarily delete
some hidden neurons, and all their connections. We forward-propagate the
input and finally backpropagate through this shrinked network.

Each time a new input batch is fed to the network, we change the subset
of hidden neurons that will be removed from the original network. In
this kind of scenario, each neuron is trying to learn the relationships
within the input data without taking for granted the help of all its
fellow neurons. Of course, the weights that are learned using Dropout
need to be reduced, when the full network will be actually used. This
compensation is due to the fact that we have learned the weights and
biases of the network in some shrinked setting, and we cannot just
re-introduce the previously removed neurons without adjusting those
weights and biases.

The probability of a neuron to be dropped out of the network is, again,
a hyper-parameter, to be chosen by the network designer. Ideally, the
smaller the drop probability, the less iterations are required to reach
an adequate accuracy level. However, by dropping neurons, the network
relies on fewer assumptions about the nature of the data than in the
original setting, and so it should overfit less.


![Dropout](/assets/images/dropout.png)

Heuristically, the Dropout operation is equivalent to a training process
that takes place by using a random set of different neural networks,
each of which is a subset of the original network. What Dropout does
when it picks up the pieces and merges the smaller networks together, is
to average the effect of the different neural networks.

#### Batch Normalization

Dropout is an incredibly clever and efficient method to address the
problem of overfitting and to speedup training. However, it introduces
yet another hyperparameter to tune somehow during validation. Of course,
this issue may seem easy to deal with, especially because it would only
require a validation step to choose the models that perform the best.

However, there are other problems that dropout does not address, and
will still affect the training performance, and consequently slow down
the learning process. These issues are mainly the aforementioned
vanishing gradient problem and parameter initialization.

We have introduced interesting solutions to these problems already, like
the use of ReLU activations to avoid saturation, and these methods are
still valid and widely used, especially together with dropout.
Nevertheless, finding a way to deal with all those issues at once would
relieve a lot of effort at training and validation time. One of the most
famous algorithm that succesfully tackles this problem is batch
normalization. [[25](references#ioffe:2015)]

Batch normalization reflects on how the distribution of the activations
at each layer changes during training, depending on every preceding
layers’ activations. This condition, called internal covariate shift,
slows down training, making it nearly impossible to learn anything
without setting low learning rates and carefully initializing the
network parameters.

Furthermore, this change in distributions makes the model need to
constantly conform to the nature of these distributions. To make the
situation even worse, the problem, similarly to the vanishing gradient,
amplifies as the network gets deeper and deeper.

The way batch normalization practically tackles the problem is by
whitening (i.e. normalizing and decorrelating) not only the network
input, but also every other layer inputs, which coincide with the
network activations. Since the full whitening at each layer would be too
computationally expensive, there are a few assumptions and
approximations to be introduced to make batch normalization applicable.
First, each feature is normalized independently.

$$\\hat{x}^{(k)} =\\frac{x^{(k)} -\\mathbf{E}[x^{(k)}]}{\\sqrt{Var[x^{(k)}]}}$$

Yann LeCun already pointed out how this kind of feature normalization
was effectively speeding up training convergence, even when the features
were correlated.[[30](references#lecun:1998)] The problem with this kind of
normalization is that it is an operation that changes what the layers
actually represent. One of the proposed solutions is to transform the
normalized inputs by scaling and shifting it back to its original space,
as follows.

$$\\hat{y}^{(k)} =\\gamma^{(k)}\\hat{x}^{(k)} +\\beta^{(k)}$$

The innovative thing about this operation is that scaling and shifting
is carried out by using trainable parameters, learned by the network
through backpropagation. Also, without this transformation, there would
be a very high chance that the representation of the learned features
was distorted.

The scaling parameters make it possible to project the normalized
features back to their original space. As a matter of fact, the scaling
and shifting parameters could be learned to represent the exact inverse
operation of normalization, if that was the optimal thing to learn for
the network.

$$\\gamma^{(k)} =\\sqrt{Var[x^{(k)}]}$$

$$\\beta^{(k)} =\\mathbf{E}[x^{(k)}]$$

$$y^{(k)} = x^{(k)}$$

We have now all the components to formally write down how batch
normalization works both in the feedforward and backpropagation steps.

$$\\mu\_B\\leftarrow\\frac{1}{m}\\sum\_{i=1}^{m}x\_i$$

$$\\sigma\_B\\leftarrow\\frac{1}{m}\\sum\_{i=1}^{m}(x\_i-\\mu\_B)^2$$

$$\\hat{x}\_i\\leftarrow\\frac{x\_i -\\mu\_B}{\\sqrt{\\sigma\_B^2 +\\epsilon}}$$

$$y\_i\\leftarrow\\gamma\\hat{x}\_i +\\beta\\equiv BN\_{\\gamma,\\beta}(x\_i)$$

As we have previously hinted, having two new trainable parameters
($\\gamma$ and $\\beta$), requires understanding how their partial
derivatives are calculated during the backpropagation phase.

$$\\frac{\\partial l}{\\partial\\hat{x}\_i}=\\frac{\\partial l}{\\partial y\_i}\\cdot\\gamma$$

$$\\frac{\\partial l}{\\partial\\sigma\_B^2}  =\\sum\_{i=1}^{m}\\frac{\\partial l}{\\partial\\hat{x}\_i}\\cdot (x\_i -\\mu\_B)\\cdot -\\frac{1}{2}  (\\sigma\_B^2 +\\epsilon)^{-\\frac{3}{2}}$$

$$\\frac{\\partial l}{\\partial\\mu\_B}  =\\sum\_{i=1}^m\\frac{\\partial l}{\\partial\\hat{x}\_i}\\cdot -\\frac{1}{\\sqrt{\\sigma\_B^2 +\\epsilon}} +\\frac{\\partial l}{\\partial\\sigma\_B^2}\\cdot\\frac{\\sum\_{i=1}^m -2(x\_i -\\mu\_B)}{m}$$

$$\\frac{\\partial l}{\\partial x\_i}  =\\frac{\\partial l}{\\partial\\hat{x}\_i}\\cdot\\frac{1}{\\sqrt{\\sigma\_B^2 +\\epsilon}} +\\frac{\\partial l}{\\partial\\sigma\_B^2}\\cdot\\frac{2(x\_i -\\mu\_B)}{m} +\\frac{\\partial l}{\\partial\\mu\_B}\\cdot\\frac{1}{m}$$

$$
\\frac{
\\partial l}{
\\partial
\\gamma}
  =
\\sum\_{i=1}^m
\\frac{
\\partial l}{
\\partial y\_i}
\\cdot
\\hat{x}\_i$$

$$
\\frac{
\\partial l}{
\\partial
\\beta}
  =
\\sum\_{i=1}^m
\\frac{
\\partial l}{
\\partial y\_i}$$

### Convolutional Neural Networks

As we have discussed in the previous sections, feedforward neural
networks are powerful, but introduce a variety of problems that need to
be addressed in order to achieve valuable and non-trivial degrees of
accuracy. The workarounds and solutions we have introduced are certainly
brilliant, and come handy in the majority of settings. However, it
sounds inefficient to try to adjust a model at all cost, especially when
the context we are trying to learn from is not really aligned with our
learning method. Let’s introduce a simple example, to understand how
powerful fully-connected neural networks are, but also where they can be
improved the most.

#### Image Classification

Suppose we need to build a neural network which needs to learn how to
classify images. A common example is handwritten digits classification.
The problem is defined as it follows: given a dataset of handwritten
digits, our model needs to be able to classify its input images
correctly.

From what we have learned in the previous sections, we can imagine to
build a neural network by modeling the input layer as a set of neurons,
each one mapping exactly one input pixel. We also need to define how to
model the output layer. Since there are ten classes of digits in total,
we model the output layer as a set of ten neurons. The final activation,
the one that connects the last hidden layer to the output layer, has to
somehow define a probability distribution of the classes, depending on
the previous activations, weights and biases. A very simple way to
achieve this, is by means of the softmax function.

$$
softmax(x\_i) =
\\frac{e^{x\_i}}{
\\sum\_{j=1}^N e^{x\_j}}$$

The softmax equation pushes the maximum value even closer to $1$, and
the other values even close to $0$. It also transforms the input space
into a probability distribution. In fact, each softmaxed value has a
value that ranges between $0$ and $1$, and the sum of all the softmaxed
inputs is always $1$.

$$\\sum\_{i=1}^N softmax(x\_i) =\\frac{\\sum\_{i=1}^N e^{x\_i}}{\\sum\_{j=1}^N e^{x\_j}}= 1$$

![Softmax Regression](/assets/images/softmax-regression.png)

A model that aims at classifying some input data by means of a
probability distribution created by a softmax function is said to do
*softmax regression*. Of course, the model we have just described is too
simple to achieve very important results. However, given the original
problem of recognizing handwritten digits, even a fully-connected
input-output network can achieve non-trivial accuracy.

Naturally, one might think that adding hidden layers to the network
automatically leads to better results. For the reasons we have discussed
in the previous sections, this is not always the case. What typically
happens, is that same levels of accuracy can be reached with a smaller
number of iterations, but the accuracy saturates anyways at sort of the
same value.

Better optimization, normalization and other methods can be useful, but
they are not a unified kind of solution. Typically, different methods
perform better in different contexts. That is understandable, and it is
useful to come to the conclusion that we might look for a different
model to solve the outer class of problem of image classification.

What we have done until now, is to model the input neurons as a straight
line of uncorrelated units, mapping each pixel of an image. That is not
the case for real images. We, as humans, understand what an image
actually represents by connecting each pixel to an area of surrounding
fellow pixels. We don’t consider an individual pixel by its own, but
rather considering it as positioned in a certain region.

Instead of arranging the input neurons as a straight line, then, we
should arrange them as the pixels are, that is composing a rectangular
bidimensional shape. Furthermore, fully-connected layers do not make
much sense anymore. In fact, it is often the case that the top-left
pixel of an image is not really correlated to the bottom-right one. We
would rather make localized connections between pixels, so that only
pixels that are close enough together are acutally connected.

Another very important idea we might consider, is that there are
repeating features within images that we would like to always be able to
recognize. Even when two very different images are inspected by someone,
the chance that similar low-level features, such as edges, or simple
curves, are shared between the images is very high. These three ideas,
that are neuron arrangement, localized connection and shared feature
recognition capabilities constitute the basis of convolutional neural
networks. Knowing that in theory will make it easier to understand why
certain architectural choices have been made.

#### Convolutional Layers

What we have just described, as always, needs to be translated into
mathematics. In particular, we need to address some questions, like: how
do localized connections translate to weights? Which kind of layers are
going to populate our network, as we lack fully-connected layers? Are
techniques like Dropout deprecated in the context of convolutional
neural networks? Let’s proceed step by step, and everything will come
together naturally.

First, we said that we are only interested in making localized
connections between our layers, because the features of the content of
the images make sense only when spatially ordered. We need to choose how
big (i.e. wide and high) the regions we want to inspect are going to be.
Suppose we choose to inspect our image with subregions of 3x3 pixels.
This means that we want to be able to recognize a certain feature in the
subregion of the input space when the hidden neuron connected to the
neurons mapping the pixels of the subregion fires. We call the subregion
connected to the hidden neuron the *receptive field* of the hidden
neuron.

![Receptive fields](/assets/images/receptive-field.png)

By sliding the receptive field of a certain step size (also known as the
*stride* of the convolution) we create a complete mapping between the
input and the hidden spaces. The most important and evident difference
between this kind of mapping and the fully-connected one is that we are
using much fewer parameters now, because the number of connections
between the input and the hidden layers depends not on the size of the
whole image, but only on the size of the receptive field.

This may seem an over-simplification, and in fact what we know until now
is only a part of the whole story. We need two more pieces of
information. First, by mapping the input to a hidden layer, we use the
same weights for the receptive field while it slides through the entire
image. We call this single shared-weighted mapping a *feature map*,
because the shared weights can be seen as a *filter* whose job is to
recognize a particular kind of feature in different regions of the same
image.

This concept is connected to what we anticipated in the previous
section, that is that we want to design our new model to be capable of
detecting low-level features regardless of the position of those
features themselves. Additionally, we allow multiple feature maps to be
created in parallel to make the vision of the hidden layer richer. Of
course, different feature maps are created by means of different shared
weights, that can be seen as different filters. With this setting, the
complexity of the convolutional step is only connected to the number of
features we would like the hidden layer to recognize within the input
space. The convolution name might be confusing at this point, but it is
actually pretty straightforward. As a matter of facts, the complete
mapping between the input and hidden layers can be seen mathematically
as follows.

$$\\sigma(b +\\sum\_{l=0}^s\\sum\_{m=0}^s w\_{l,m} a\_{j+l, k+m})$$

The equation contains various terms we have already seen. $\\sigma$ is
the activation function, responsible for determining if the hidden
neuron will fire, given the weighted sum of the connections with the
input space. In this case, by making localized connections by means of
input subregions of size $s$x$s$, the weighted sum is mathematically
written as a convolution operation.

#### Pooling Layers

Convolutional layers are not the only type of layer that define how a
convolutional neural network works. As we stack convolutional layers on
top of each other, we find out that the complexity of our model (i.e.
the number of its parameters) grow very fast.

However, there are a lot of information that can be discarded after the
convolutional step. In fact, it is true that we want to leverage the
spatial information of the input, but it also true that we can drop some
of that spatial information after features are detected. After all, it
is important to know where the feature is, but we can make a positional
approximation as the network grows, trying to compose low-level features
into high-level ones. Pooling layers are used exactly for this purpose.
We want to drop some of the spatial information for the sake of model
simplicity.

Again, we need to translate this concept into practice. Let’s choose
another subregion, this time in the hidden space. This subregion
analyzes the value of each activation, and applies a function in order
to preserve the most important information in the region. The most
common function is the $max$ function, that simply picks the maximum
activation and drops all the others. However, in principle every
function that makes sense in preserving information can be used as a
pooling function.


![Max-pooling](/assets/images/max-pooling.png)

By putting convolutional and pooling layers together, we are mostly done
with the design choices that make convolutional neural networks more
powerful in contexts where spatial information is important.


![Convolutional and pooling layers](/assets/images/cnn-layer.png)

Convolutional neural networks are mostly made of a long stack of
convolutional and pooling layers. However, especially for classification
problems, fully-connected layers and softmax are still used to create
the final probability distribution of the output. As a consequence, even
if convolutional layers reduce the complexity of the model, relieving
the model from heavy normalization techniques, all we have learned by
analyzed vanilla neural networks is still valid. The logic, as always,
is that rather than design the network by trial-and-error, it is always
useful to analyze its layer-by-layer behavior, and to spot inconsistency
or other issues analytically.

### Recurrent Neural Networks

In a wide variety of tasks, the context is really important to spot the
right meaning of an information. For instance, since many words in
natural languages have an ambiguous pronunciation, our brain
automatically find the right word associated to a sound just by using
the information from the context. Recurrent Neural Networks are networks
with a special architecture, designed exactly to leverage context
information. They do so by using a time-wise computation, so that the
previous states of the network influences the subsequent states. The
influence of the network at time t to the network at time t+1 can be
provided by the previous outputs, the previous state of the weights or a
combination of both. To represent this influences, there are used the
so-called computational graphs.

The problem of vanishing gradient becomes even more problematic in
Recurrent Neural Networks, because the effect of states far in time can
be very small. To balance this vanishing effect, architectures like Long
Short Term Memory (LSTM) are now really popular. The idea behind LSTM is
to use gates to decide which information to filter, and which one to
send to the next steps of the computational graph.. The filter is
learned, as well as the other parameters.

LSTM are today solving a variety of problems, even ones in which time is
apparently not directly involved. In fact, in every domain in which the
prediction can be improved gradually, Recurrent Neural Networks can be
useful.

### Autoencoders

The last architecture we want to analyze is autoencoders. Autoencoders
are neural networks which are mainly used in unsupervised learning
settings, to face problems such as dimensionality reduction through
encoding and decoding an input.

Autoencoders operate in a relatively easy way. First, they try to
extract features from a particular input set (the encoding part). They
do so by means of a deep stack of layers (mostly convolutional and
pooling layers), until the dimensionality is reduced to a minimum. So,
the encoder simply maps the input to a code, i.e. a set of features that
represent an encoded version of the input data. Finally, they try to
reconstruct the input data from the code generated by the encoder (the
decoding part).

Typically, the decoding task is carried out by a symmetric stack of
layers, each one of them corresponding to a layer in the encoder. For
instance, a convolutional layer in the encoder corresponds to a
deconvolutional (or transpose convolutional) layer in the decoder; a
pooling layer in the encoder corresponds to an unpooling layer in the
decoder.


![Autoencoders](/assets/images/autoencoder.png)

Analytically, it is possible to represent what shown by the figure above
as follows:

$$\\phi:\\mathcal{X}\\rightarrow\\mathcal{F}$$

$$\\psi:\\mathcal{F}\\rightarrow\\mathcal{X}$$

$$\\phi,\\psi =\\arg\\min\_{\\phi,\\psi}\\|X - (\\psi\\circ\\phi) X\\|^2$$

While it may seem autoencoders are simply architectures that put basic
layers together, there are some complications that make them more
interesting than expected. Since the decoder part of the autoencoder
acts as a generative model, it runs into problems that are typical of
generative networks, such as patterned results, unpooling inaccuracies,
and so on. All these problems in upsampling can be traced back to the
way neural networks map layers during deconvolutions.
[[41](references#odena2016deconvolution)]

---
title: Abstract
---

Practical applications and theoretical results in computer vision have
dramatically improved since the massive utilization of neural networks.

In this thesis we face the problem of retrieving quantitative
information of objects, i.e. volume and weight, from pictures. We focus
on the food ingredients domain. The goal is an application that takes as
input one or multiple pictures of ingredients, and is able to provide a
variety of suggestions about them, like weight, calories, and so on.

To solve this problem we build a fully-convolutional pipeline of neural
networks, that goes from the pictures of ingredients to the
reconstruction of their volume. We choose volume as the main measure to
obtain among other quantitative information. We use a convolutional
encoder-decoder architecture to solve the subproblem of classifying the
objects in a picture and separating them from the background. We feed
the images preprocessed this way to a convolutional-recurrent network,
to solve the problem of 3D shape reconstruction. This network can
predict the 3D shape of an ingredient even from a single picture as
input, thanks to its knowledge of ingredients learned during training.
We then devise a method to obtain the absolute value of the volume,
starting from the relative one and the reconstructed volume of an object
of standard size in the real world, used for comparison.

To solve the problem of having enough labeled data in a scalable way, we
study the utilization of 3D synthetics models, by creating a novel
synthetic dataset. Thus, we provide some experiments to study the power
of networks trained with synthetically generated images to predict real
ones. We study the influence of backgrounds, light conditions, textures
and camera orientations on the predictive performance. Thus, we gain
insights on how to properly build a synthetic dataset.

We provide a functioning pipeline that solves the aforementioned
quantification task. Finally, we also propose a novel multitasking
architecture.

This thesis was supervised by Stefano Ceri and Marco Brambilla from
Politecnico di Milano, and Pavlos Protopapas from Harvard University. We
spent three months in Cambridge, where we also presented two workshops
on deep learning at Harvard ComputeFest 2017.

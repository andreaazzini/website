---
title: Experiments
---

In this chapter, we will take a closer look to the experiments we have
performed to verify the performance and accuracy of our models. We will
show the results of our experiments in detail, paving the way for a
final discussion on the results themselves, and how they could be
improved in the future.

For both the segmentation and 3D reconstruction experiments, we have
implemented our own Python libraries and scripts. The most relevant
framework we have leveraged to achieve our goals is TensorFlow.
[[1](references#abadi2016tensorflow)] TensorFlow is an open source software library for
numerical computation using data flow graphs. Nodes in the graph
represent mathematical operations, while the graph edges represent the
multidimensional data arrays (tensors) communicated between them. The
flexible architecture allows you to deploy computation to one or more
CPUs or GPUs in a desktop, server, or mobile device with a single API.
TensorFlow was originally developed by researchers and engineers working
on the Google Brain Team within Google’s Machine Intelligence research
organization for the purposes of conducting machine learning and deep
neural networks research, but the system is general enough to be
applicable in a wide variety of other domains as well.

As we have mentioned in Chapters 2 and 3, the encoder-decoder architectures we
have built require a high degree of computational power and parallelism
to leverage the potential of convolutional layers as much as possible.
The typical processing units that are used in this kind of setting are
GPUs. In particular, the GPUs we have used are:

-   **Nvidia GeForce GTX 1080**, provided by the Harvard Institute for
    Applied Computational Science (IACS), which we have used during our
    period as guests at Harvard University

-   **Nvidia Tesla K80**, within a Microsoft Azure virtual machine
    provided by Politecnico di Milano DEIB, which we have used for the
    experiments we have run in Italy

Without this kind of environment, our work would not have been possible.
To give a comparative idea, 10,000 training iterations of the
MiniAutoencoder we have described in Section 4.1 would
have required nearly 48 hours to complete without a GPU. With the help
of the GPUs, the training time reduced to 3 hours approximately, making
our experiments feasible.

### Segmentation

As we have mentioned in the previous sections, the way we have
structured the segmentation experiments as shown in table
\[table:experiments\]. For each experiment, we have used the same
autoencoder, which has been trained for 10,000 iterations. We will
report the plots for training accuracy and loss function, along with the
result of the segmentation on real images. We will not report a test
accuracy here, since we do not possess the segmentation labels for each
test image, and because the segmentation result is just intermediate in
our case.

These experiments try to go beyond the verification of accurate
segmentation. The objective of the experiments are:

-   Object segmentation (i.e. shape and contour definition)

-   Object classification

-   Synthetic to real images generalization

We will keep track of the quantitative and qualitative results for each
of those categories, and elaborate how we can solve the problems that
occur, if there are any.

#### Experiment 1: Single objects, monochromatic backgrounds

The first experiment involves a very simple training set, made of
synthetic images representing food with a very simple interpolation of
different textures. The input images represent a single object, with no
complex shadows or lights and monochromatic backgrounds. The setting is
very simple to test that the segmentation autoencoder is working very
well on a trivial problem. It is also interesting to verify its
preliminary generalization capability against a test set of real
images.


![Experiment 1 training set](/assets/images/01_training_set.png)

As it is noticeable from the accuracy plot below, after 10,000
training iterations, the training accuracy reaches a value of 99%, which
looks very high, considering the problem of pixel-wise semantic
segmentation. This result was definitely expected, given the trivial
nature of this first experiment.

![Experiment 1 training accuracy](/assets/images/01_train_accuracy.png)

The learning task seems to work as expected, so we test it against real
images. Here, we report the results of real test sets only, because we
are more interested in the verification of the generalization property
than in the testing accuracy against other synthetic images, which we
expect to be very high as well.

![Experiment 1 test results](/assets/images/01_test_set.png)

The figure above shows how the shape and contour of the objects
are determined correctly, while the coloring (i.e. the class the object
belongs to) is not. This may be due to the fact that multiple features
are encoded similarly, and the decoding process cannot reconstruct them
assigning them to unambiguous class. This is a very important hint,
which tells us how the classification process may need to be delegated
to another entity, and cannot be generalized, when using such training
set. We will further investigate the classification generalization in
the following experiments.

However, this first experiment gives us a lot of positive insights
already. First of all, the autoencoders we have built seem to work
properly. Secondly, the segmentation of an object itself is
generalizable for this simple single object image set. However, we have
seen how the labeling, corresponding to a sort of classification, cannot
be generalized, and needs to be addressed in a different way.

#### Experiment 2: Single objects, random backgrounds

The second experiment includes a slight complication. The training set
is still based on synthetic images of food. The input images portray a
single object, with no complex variation of its features, and random
background images. The random backgrounds have been generated
considering the use of colors that are close to the ones of the depicted
objects.

The experiment aims at finding out if, with more complex backgrounds,
the network is focusing more on the features of the objects, discarding
everything different, such as the color of the background.

![Experiment 2 training set](/assets/images/02_training_set.png)

The accuracy plot shows how, after 10,000 training iterations,
the training accuracy reaches a value of 98%, which is slightly lower
than the one of experiment 1, but still very high. We need now to verify
its generalization potential.

![Experiment 2 training accuracy](/assets/images/02_train_accuracy.png)

![Experiment 2 test results](/assets/images/02_test_set.png)

The figure above shows similar results to the experiment 1 counterpart.
It is clear how the shape and contour of the objects are determined
correctly, while the coloring is not. Furthermore, in this experiment,
we notice how the network has not been able to learn to recognize and
segment eggs correctly. This may be due to the fact that training images
have a background which is too similar to the object itself, especially
for white eggs. The network is still not able to recognize them after a
cycle of 10,000 iterations.

Again, even though the segmentation is generalizable, the classification
is not. Nothing has really changed quantitatively, but it is evident how
the test results look worse in this experiment than in the first one.
The analysis on why this is the case is quite intuitive. By forcing the
network to focus on the precise features of training images, we are
overfitting the training data. In fact, the training accuracy and loss
look identical, but the test images, representing how the model is able
to generalize, look worse than expected. We will still study the same
difference for multiple object, by keeping in mind which insights this
experiment has given us.

#### Experiment 3: Multiple objects, monochromatic backgrounds

This experiment changes the context, studying the impact of a different
set of objects, which belong to the same classes as before, coming in
multiple entities. The input images represent multiple objects belonging
to the same class, with no complex shadow or light, and monochromatic
backgrounds. The experiment aims at finding out if the network is able
to distinguish multiple instances of objects within the same image.

![Experiment 3 training set](/assets/images/05_training_set.png)

The accuracy plot below shows how, after 10,000 training iterations,
the training accuracy reaches a value of 96%, which is slightly lower
than the ones in the single object experiments, but is still high. We
need now to verify how it generalizes to real images.

![Experiment 3 training accuracy](/assets/images/05_train_accuracy.png)

![Experiment 3 test results](/assets/images/05_test_set.png)

The figure above shows how training is not stable anymore. Even
though the loss function keeps decreasing, the training accuracy does
not get significantly better over time. There is a high chance of
overfitting in this scenario, since we are forcing the model to adapt to
the training data, and we can see basically no improvement and a lot of
noise after 10,000 iterations.

The test images look worse in this multiple context than in the previous
single settings. The generalization capabilities, in this case, have
been decreased dramatically. It is still worth trying to complicate the
training set and see if anything improves at all. As always, delegating
the class selection to segmentation autoencoders will not work in this
scenario.

#### Experiment 4: Multiple objects, random backgrounds

Similarly to what we have done in the single object setting, we want to
verify if, by complicating the backgrounds of our images, the model is
forced to understand the true nature of the object themselves, thus
learning to recognize and segment them better. We have seen how this was
not working as expected for single objects, because of the synthetic
nature of the training images. By forcing the network to focus on the
synthetic features, the model should get worse at generalizing to real
images.

![Experiment 4 training set](/assets/images/06_training_set.png)

The accuracy plot below shows how the training accuracy reaches a
value of 90%, which is surprisingly bad after 10,000 iterations,
especially comparing it with the accuracy of the same network in the
previous experiments. It is clear that the new model is presenting the
same problems we have noticed during Experiment 2.

![Experiment 4 training accuracy](/assets/images/06_train_accuracy.png)

![Experiment 4 test results](/assets/images/06_test_set.png)

The figure above shows how the training process is presenting the
same problems we have seen in Experiment 2,
that are high variance and poor generalization power. In fact, the test
shows how the segmentation process, even when grasping the true shape of
the objects, is not as accurate as the single object case. Additionally,
we have a further confirmation of the fact that complicating backgrounds
to make the network focus on the objects’ features is not a good way to
improve the segmentation task, as described by both training and test
accuracies. We stop using this method for the following experiments.

#### Experiment 5: Single and multiple objects, monochromatic backgrounds

After the first four experiments, we can draw some conclusions already.
First, it looks like complicating the training set adding noise instead
of simple backgrounds is not working as a method for the improvement of
the learning task. Second, single objects seem to be much easier to
recognize than multiple objects. These aspects bring us to a question:
what if we tried to train the network using both single and multiple
objects? Would the training process be harder, or the generalization
capabilities of the network compromised? In order to answer these
questions, we put together the dataset of Experiment 1 and Experiment
3.

![Experiment 5 training accuracy](/assets/images/all_train_accuracy.png)

The plot above shows how the training accuracy reaches a
value of 96%, which improves the results of the multiple objects alone
setting. The tests, as shown in the figure below, prove that good
results can be generalized to real images only, both for single and
multiple objects. However the generalization power varies a lot,
depending on the nature of the shadows, backgrounds and other
complications. This aspect suggests that, even if training single and
multiple objects together may be a good idea for keeping the
segmentation stable, it is absolutely crucial to introduce some
variations in the training set to make the synthetic images closes to
their real counterparts.

![Experiment 5 test results](/assets/images/all_test_set.png)

#### Experiment 6: Single and multiple objects with complex variations

From all the previous experiments, we can notice that the segmentation
of the objects is overall working, while the generalization on real
images is not affected by background diversity in the training images.
However, we know real images are definitely more complex in features
such as texture, lights, shape, and so on. It is worth trying to
introduce new variations in the training set, to capture the majority of
characteristics that make the generalization problematic.

With this objective in mind, we can introduce a new training set, which
is made of images of objects belonging to any class, both single and
multiple. The new variations include new textures, more complicated
light conditions, shadows, and very simple backgrounds. The new training
set is made of 25000 images for the same 5 class setting. With this
experiment, we want to reinforce the generalization power of the model,
and we ignore the classification aspect of the segmentation, labeling
all the objects in the same way.

![Experiment 6 training set](/assets/images/final_training_set.png)

![Experiment 6 training accuracy](/assets/images/final_train_accuracy.png)

After 10,000 iterations of the training task, we can see how the model
is trying to minimize the loss function without being really able to
improve the overall accuracy of the model anymore, which reaches its maximum value at 80-85%. It
is immediately noticeable how the new training set made the training
accuracy lower than the previous experiments. However, the results of
the test are definitely more accurate in this new scenario.

![Experiment 6 test results](/assets/images/final_test_set.png)

The figure above shows the major improvement of the
generalization task, made possible by the introduction of more
complicated variations in the training set. The features of the objects
seem to be way more clear for the model in this setting, because shadow
are now excluded from the segmentation, and objects are detected with
their right shape even when the background is not white. This
experiments would be enough to make the pipeline work properly, but it
also paves the way for a final test, which tries to generalize well on
the segmentation task, while not being classification agnostic.

#### Experiment 7: Mixed objects with complex variations

With experiment 6, we proved not only that our autoencoder was able to
segment the objects accurately, but also that the segmentation task was
generalizable. To do that, we dropped one of our objectives, which was
to prove that the classification power of the model was generalizable,
too.

Experiment 7 is aimed at overcoming this issue. We created a final
synthetic training set, made of images containing different food in
different textures, shapes, light conditions, camera angles, and all the
variations we have already considered during experiment 6.

![Experiment 7 training set](/assets/images/mixed_training_set.png)

![Experiment 7 training accuracy](/assets/images/mixed_train_accuracy.png)

After 10,000 iterations, the model has definitely learned how to segment
and classify the objects in the training set, with a training accuracy
reaching 99% (see accuracy plot above). This training set was the
only one which contained mixed objects together in the same pictures, so
the training accuracy plot shows how this further complication does not
affect the learning task.

![Experiment 7 test results](/assets/images/mixed_test_set.png)

Finally, the figure above shows the results of our model when
tested against real images. What is immediately observable is that,
despite the further complications of the synthetic training set, not
only the power of segmenting the objects is retained, but also the
overall pixel-by-pixel coloring actually mirrors the true class of the
objects themselves. It also reveals the potential of
classification generalization during the segmentation task, which is a
very powerful ability for our autoencoder. In fact, even though the
classification results are not perfect, it is evident how, differently
from the previous experiments, the model is now starting to understand
which object it is segmenting.

### The Bridge between Segmentation and 3D Reconstruction

Before looking at the experiments about the 3D reconstruction, we need
to clarify how we build a bridge between the two modules. As we have
seen, the output of the segmentation module is a segmentation map, a
224x224 image which is colored depending of the nature of the objects
inside the image. However, the input of the 3D reconstructor is a
127x127 image, representing a real image, with a white background and
the real object in the foreground. We need to switch between the two
contexts.

![The bridge between the segmentation and the 3D reconstructor modules](/assets/images/bridge.png)

The figure above shows how the input images are first segmented,
and then filtered so that what the segmentation module thinks belongs to
the background is completely whitened. As we will see in the following
section, this operation is necessary to make the input perfect for the
3D reconstruction module.

### Volume Reconstruction

In this section, we review four kinds of experiments. The first one is
aimed at proving the contribution of our dataset, and that we found a
very efficient way to build it so that it may generalize to unseen real
images. We test it against the ShapeNet benchmark. The second is aimed
at proving the improvement given in our pipeline by the segmentation
step. To do so, we test the results of the volume reconstruction with
and without the previous segmentation. In the third experiment, we test
our method of using a standard sized object to obtain the absolute value
of the height and width of the ingredients.

To do so, we test if the ratio between the reference object and the
ingredient, that we randomly vary in our training set, is respected
during the prediction. The last experiment is about achieving
quantification by comparison, by leveraging two 2D images obtained with
the same input, one with the Stabilo and the object and the other with
the object alone. In all these experiments, we show the results for the
*pear* class, because in our setting pears are the objects that come
with a more diverse set of variations in many features, such as shape,
size and color.

#### Experiment 1: Novel vs Benchmark

Here we show the contribution of our dataset to the final predictions.
The first figure below shows some pictures from the Como test set, which is
the test set we created by taking some pictures of food ingredients.
The second figure below shows the filtered input images, as we used in this
test.

In fact, here we wanted to test the best results obtainable in an ideal
case, while in the next experiments we focus more on the images
segmented with our SegNet. The third and fourth figures below show
the prediction of the network trained with different datasets.
Furthermore, the table below shows the different loss functions.

  |             | ShapeNet   | ShapeNet + Novel   | Novel   |
  | ----------- | ---------- | ------------------ | ------- |
  | Mean loss   | 0.4        | 0.05               | 0.8     |

![Some views of a pear from the Como test
set.[]{data-label="fig:pears1"}](/assets/images/pears1.png)

![Some views of a pear from the Como test set cleaned from the
background.[]{data-label="fig:pears2"}](/assets/images/pears2.png)

![Prediction on the cleaned images after training with our training
set.[]{data-label="fig:pred1"}](/assets/images/pred1.png)

![Prediction on the cleaned images after training with
Shapenet.[]{data-label="fig:pred2"}](/assets/images/pred2.png)

The images in the test set come from the Como test set, and are not
present in the training set. The comparison shows a few insights. Even
though the model we trained on ShapeNet has a decent behaviour on the
segmented images, it is definitely outperformed by the model we trained
through our synthesized training set. This shows that the way we used to
build our dataset can be generalized and used for more classes.

The model which was trained with our synthesized training set recognizes
even small details, such as very local peculiarities in shape, and it is
also able to reconcile those peculiarities from different views. What is
also observed is that the pretrained model with ShapeNet is helpful when
followed by a fine-tuning with our classes, as it happens in similar
computer vision tasks.

#### Experiment 2: Pre-Segmentation vs Non Segmentation

Here we test the contribution given by the segmentation step. We use two
pears views from the Como test set as input, and we test the volume
reconstruction by feeding the reconstruction network with either one or
both of them. The first and third figures below represent the
inputs, while the second and fourth figures below represent the
predictions on the unsegmented and the segmented inputs.

![From left to right: Como data set picture, its segmentation with
SegNet, the Segnet mask
predicted.[]{data-label="fig:perePre1"}](/assets/images/perePre1.png)

![Prediction on the non segmented vs segmented input with 1
view.[]{data-label="fig:with1"}](/assets/images/with1.png)

![From left to right: Como data set picture, its segmentation with
SegNet, the Segnet mask
predicted.[]{data-label="fig:pre2"}](/assets/images/pre2.png)

![Prediction on the non segmented vs segmented input with 2
views.[]{data-label="fig:with2"}](/assets/images/with2.png)

Here the improvements given by the segmentation phase are very
considerable. The images reconstructed from the unsegmented views look
almost random. On the other hand, the images predicted out from the
segmented input show interesting properties.

First, it is confirmed the property of the network to gain further
information from subsequent images, property absent in this case with
unsegmented images. Moreover, the predictions show a very nice property:
they are robust with respect to small errors in the segmentation. For
instance, the shadow in the first image is ignored at prediction time.
This nice property is gained because, after being trained on a lot of
pears, the model has learned that such kind of extension is not a
feature of pears. This result is not obtained when the network is tested
after being trained only with the ShapeNet training set. In such case,
the shadow causes an error in the prediction.

#### Experiment 3: Proportions

As described in the previous section, here we test the capability of the
network to reconstruct two objects by keeping the real ratio between
their volumes. For these experiments, we enriched the previous training
set with pictures of the object close to a Stabilo, as in
the figure below, by varying their relative sizes, as shown in the
pictures below:

![Two pictures from the training set for proportions
learning.[]{data-label="fig:trainprop"}](/assets/images/train_proportions.png)

The proportions are perfectly learned during training. In the figures below
we show an example of how the proportions are
respected with two test images, that come from another test set that we
generated.

![Objects in the proportion test
set.[]{data-label="fig:testprop"}](/assets/images/test_prop.jpg)

![Smaller pear in the test
set.[]{data-label="fig:prop1"}](/assets/images/prop1.png)

![Bigger pear in the test
set.[]{data-label="fig:prop2"}](/assets/images/prop2.png)

By counting the Stabilo voxels in width in these predictions, and use
the proportion between these voxel and the ones in the pears, we can
obtain a first quantification estimation:

  |          | Real height (cm)   | Predicted height (cm)   |
  | -------- | ------------------ | ----------------------- |
  | Pear 1   | 7.5                | 7.7                     |
  | Pear 2   | 13                 | 12                      |


  |          | Real width (cm)   | Predicted width (cm)  |
  | -------- | ----------------- | ----------------------|
  | Pear 1   | 5.8               | 6.4                   |
  | Pear 2   | 7                 | 8.33                  |

#### Experiment 4: Absolute Volume and Quantification

In this experiment, we finally retrieve the absolute volume of objects
in a way that is fully automatic. In fact, the voxel proportion is not
counted manually anymore, but by difference in reconstructed models. The
procedure is described in the previous section, but we remember here
that a script computes the ratio between the Stabilo and the pear in the
segmented image. Then two reconstructions are made, the one with the
Stabilo and the one without it. The one without the Stabilo is scaled
according to the ratio found before, and then, by subtracting the voxels
from the complete reconstruction, the voxel quantity associated with the
Stabilo is obtained. From this measure, all the others are obtained by
proportion. We show an example from a pear in the test set above.
Its reconstructions with and without the Stabilo
are shown in the figure below.

![Reconstructions with and without the Stabilo](/assets/images/reconStabil.png)

We approximate the volume of the Stabilo as a parallelepiped, of sizes
$10cm \\times 2cm \\times 1cm$, which results in an approximated volume of
$20 cm^3$. We weighted the pear, which was approximately $350 g$. The
specific weight of pears is $3.5 \\frac{g}{cm^3}$. Therefore, the volume
of the considered pear is $100 cm^3$. The ratio between the Stabilo and
the pear height in the picture is $0.9$. The voxels in the
reconstruction of the Stabilo plus the pear are $38,144$, while the
reconstruction of the pear alone has $35,064$ voxels. The scaled
reconstruction with the pear alone has $31,558$ voxels. The difference
between it and the complete reconstruction is $6,587$ voxels, which are
consequently the voxels assigned to the Stabilo. Therefore, $6,587$
voxels correspond to a volume of $20 cm^3$. The ratio
$\\frac{volume}{voxels}$ in this case is $329.35$. Therefore, the volume
of the pear is $95.81 cm^3$. Very close to the actual volume,
$100 cm^3$. The pear has $5.7 \\frac{cal}{g}$, therefore for our pear we
get:

  | **Predicted volume ($cm^3$)**   | **Actual volume ($cm^3$)**   |
  | 95.8                            | 100                          |
  | **Predicted weight ($g$)**      | **Actual weight ($g$)**      |
  | 335.3                           | 350                          |
  | **Predicted calories**          | **Actual calories**          |
  | 191.1                           | 199.5                        |

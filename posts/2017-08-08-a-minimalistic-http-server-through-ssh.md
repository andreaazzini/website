---
title: A minimalistic HTTP server through SSH
date: 2017-08-08 19:48:00
tags: ai,deep-learning,computer-vision,ssh,tensorboard
---

It's 2017. Working in the field of AI, especially if you are in computer vision, there is a high chance you are using a GPU-powered server via `ssh` to train your models, and that you are dealing with thousands, even millions of training images.

You probably manipulate (e.g. scale, crop, flip, stack) your images on a daily basis, and you need to make sure that the operations you are performing are _correct_.

So how do you check your pictures without copying them from and to your remote server? How do you do that without wasting time? I propose a minimalistic solution here, which leverages the `LocalForward` option provided by `ssh`.

Let's say our remote machine (`remote`) can be found at `remote_ip`, and analogously our local machine (`local`) has can be found at `local_ip`.
On `local`, edit `~/.ssh/config` as follows:

```
Host remote
    HostName remote_ip
    LocalForward 18080 127.0.0.1:8080
```

On your `local` terminal, you can now `ssh username@remote` to access `remote` with the previous forwarding rule. When you're there, just start an HTTP server. If you use Python 3, you could type `python -m http.sever`, but any simple server listening to port 8080 would do the job.

Back to `local`, you can access your HTTP server on a browser, navigating to http://localhost:18080.

Pretty simple, right?

You can definitely use multiple `LocalForward` rules to access a different service on a different port (i.e. Tensorboard).

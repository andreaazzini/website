---
title: The simplicity and power of Elixir: the ws2048 case
date: 2015-10-12 19:11:00
tags: elixir,game,otp,genserver,ws2048
---

I was first introduced to Elixir by my colleague and friend Aleksei Magusev —[@lexmag](https://github.com/lexmag). As a student with some basic Erlang knowledge, I was immediately fascinated by the simplicity and elegance of Elixir, especially being aware of the power of the Erlang building blocks, such as the Erlang VM, the OTP framework, and so on. I spent most of my time getting into its vision and philosophy, trying to distance from the object-oriented mindset that my previous studies and experiences had consolidated. At the same time, Aleksei was continuously feeding me with nice and inspirational ideas to make me improve. One of the coolest was a WebSocket implementation of his tty2048, a terminal version of the notorious game 2048. The idea seemed very valuable to me, because I could have got my hands dirty with the rising Phoenix web framework. I came up with a cooler one though: a collaborative version of 2048, in which each player could democratically and simultaneously choose a direction to move towards, given that the most voted one would have been chosen at the end of a timeout. The inspiration came from the Twitch Plays Pokémon social experiment, something I was really into in 2014, when first went viral. Such a simple idea can still be very interesting to give birth to, because of the key concepts in modern web programming it incorporates. Before we start looking at some code, do you think this game could fit in something around 150 lines of code, without compromising on expressiveness and simplicity? 😎

## Metaprogramming and Phoenix  channels

I am intimidated by this title as well, but trust me, we shouldn’t be. Let’s look at some code!

<script src="https://gist.github.com/andreaazzini/edc680b479aafae3d5ca.js"></script>

This is the simplicity of Phoenix channels. Even without knowing how the Ws2048.Move module looks like, it’s kind of easy to understand what’s going on here. The functions in GameChannel are simply callback implementations of the Phoenix.Channel behaviour. Just to sum them up, join/3 aims at establishing the connection; terminate/2, on the contrary, handles its termination; handle_info/2, similarly to its GenServer and GenEvent equivalents, handles special messages — in this case, the reception of the :peek_grid signal, which is sent when the channel has been joined (line 7), and it’s important to peek the game state when you first access the game; handle_in/3, finally, manages the reception of socket messages, which are continuously broadcasted whenever a game event is detected — the way this event detection occurs is gonna be explained in a short while.

Along with the simplicity and straightforwardness of the Phoenix APIs, the expressiveness of the Elixir language is shown here within the context of its extremely powerful code generation features. Lines 18–25 aim at generating four different handle_in functions. As you can see, metaprogramming is as simple as standard functional programming concepts in Elixir. Here, since the four functions share the same logic, it is very easy to embody them and use unquote when a term is meant to be used as part of code generation.

## The game

Awesome, but how does the Ws2048.Move module look like? Well, it’s modelled as a simple GenServer, and the calls in the GameChannel are nothing more than GenServer calls, handled by — guess what? — handle_call/3.

<script src="https://gist.github.com/andreaazzini/73f942a7afc5e4b53f44.js"></script>

Everything seems already pretty interesting, but it lacks the connecting logic with the core of the game. This is when tty2048 plays its role. At this moment, the only thing we have done is collecting moves from the users that connect to our channel. Tty2048 is provided with a Game GenServer, that initializes a GenEvent manager. This manager gets notified each time a move is performed and is in charge of notifying the watchers about any kind of occurring event. In the end, we currently need to do two things: actually performing the move, and building a watcher that is able to react to game events.

## Tty2048 integration

To accomplish the first task, let’s add some functions to the Ws2048.Move GenServer.

<script src="https://gist.github.com/andreaazzini/b998b1b46b56c876117d.js"></script>

The init/1 and handle_info/2 functions, which are, again, GenServer behaviour callbacks, handle the process timeout. In particular, the former makes the first tick — that is, sending itself a message and starting a timer — while the latter makes a decision at each tick, and restarts the timer. So, each time the time runs out, the Move GenServer reacts by calculating the direction the majority of users have voted for, and notifying tty2048 about the decision. But who is reacting to actual changes in the game board, or even to game over signals? To do this, we need a watcher that handles the aforementioned game event logic.

<script src="https://gist.github.com/andreaazzini/5ba70d3884b38744af0f.js"></script>

The watcher only needs an init/1 function to perform an action as soon as the game is initialized, and some handle_event/2 functions to handle game events. In this case, the game events we are dealing with are board moves and game over. In particular, the game over event gets back to our beloved Move GenServer, which is in charge to restart the game.

## A simple but meaningful supervision tree

Let’s to this elegantly by creating a supervision tree with a dedicated game supervisor that could be terminated and restarted without influencing the other children.

<script src="https://gist.github.com/andreaazzini/79eb24c0348231d64f3e.js"></script>

Doing as we said above is just about changing the main supervisor’s children structure, and adding a few lines containing the initialization of our new game supervisor. Now we can finally restart the game with no harm when no move is possible, just by adding the following short logic in our Move module.

<script src="https://gist.github.com/andreaazzini/717ff30a86944b0d8c72.js"></script>

## That’s (almost) all  folks

This is actually far from being all, but it is what we need at backend to handle the original game idea logic. This article is trying to make you realize how powerful Elixir and Phoenix are, without spending too many lines on tedious technicalities. You can get the complete source code on ws2048 GitHub page, and even play the game of course! Thank you for reading, I truly hope you enjoyed! 🍻

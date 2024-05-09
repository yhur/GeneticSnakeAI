# Genetice SnakeAI 
This repo is a modification of Chrispresso/SnakeAI, https://github.com/Chrispresso/SnakeAI.

![image](https://github.com/yhur/GeneticSnakeAI/assets/13171662/cb5e169b-ebe6-442b-ad81-e648da765c30)

Thanks to Chrispresso/SnakeAI, it was easy in explaining the Genetic Algorithm to my student. And while teaching them with it, I wanted to add some minor features for even easier explanation.

The original snake_app.py is intact and instead mysnake.py is cloned from it and modified for

1. saving and loading the learned weights
2. running it without displaying to expedite the learning process
3. step excution to explore and explain the input and the neuron states

The usages is as follows
<pre>
Usage: python3 mysnake.py [OPTIONS] [CMD]

Options:
  -w, --weights TEXT  Weights Data Folder
  -h, --help          Show this message and exit.

Command:
  show
</pre>

1. The weights are loaded from and stored in the default folder `weights`, if the -w option is not specified. Or if you want a particular folder to keep the weights, you can run it as follows. 
<pre>
    python3 mysnake.py -w myweights
</pre>

2. It has only one command `show`. When invoked with it as below, it will show the snake movement on the screen, and if without, it will not show the snake movement and the learing process is faster.
<pre>
    python3 mysnake.py show
</pre>

If you want to run the learning faster, then you can run it without the `show` command as below.
<pre>
    python3 mysnake.py
</pre>

3. You can pause the program by pressing `s` or `space` on the game canvas, and explain the input nodes and the neurons on each layers.
<pre>
  keystroke 's' or 'space' for step execution
  keystroke 'r' for resuem
  keystroke 'escape' to quit
</pre>

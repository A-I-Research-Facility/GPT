# GPT

## About

This is a simple decoder only model based on the research paper "Attention is all you need". We create a Bigram language model, that utilizes Multi Head Atterntion and is able to generate some text based on the input text file. The model is not created for one-on-one conversations, it simply generates new words and sentences based on the language of the input file.

## Requirements

The following libraries must be installed for running this model :

1. Pytorch
2. Cuda

Apart from the required libraries, it also important to run this model on a dedicated GPU as we work with almost 1 million parameters. This is nothing compared to the 145 billion parameters that are used by OpenAI GPT-4, but this still requires dedicated GPU for training. If for some reason you want to use CPU, then modify the code and change the device to 'cpu' in hyperparameters.

## Running the model

To run the model, clone the repo, and in the same directory, create a text file called "input.txt". In that text file, copy and paste large amount of text data, like a novel or a play.

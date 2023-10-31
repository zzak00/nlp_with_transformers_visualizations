# NLP with transformwrs chapter 1 :

Language is at the core of human communication, and for centuries, we've endeavored to unravel how it is intricately woven into our cognitive processes. Today, we are on a quest to advance further by introducing computers to the realm of human language through **Natural Language Processing (NLP)**. NLP is an artificial intelligence branch dedicated to harnessing machine learning techniques for a wide range of language-related tasks.

In recent years, NLP has witnessed a remarkable evolution with the emergence of transformative techniques such as **transformers**, the **encoder-decoder framework**, and the **attention mechanism**. These innovations have reshaped the landscape of NLP. To gain a deep understanding of these developments, one of the most valuable resources is the book **Natural Language Processing with Transformers: Building Language Applications with Hugging Face**, which we are summarizing in this series of articles.

## The Encoder-Decoder Framework

The concept of an encoder-decoder, also known as a **sequence-to-sequence architecture**, made its debut in 2014 with the publication of the "Sequence to Sequence Learning with Neural Networks" paper by a Google research team. This architecture finds its forte in scenarios where both the input and output entail sequences of arbitrary lengths.

Comprising two core components:

- **The Encoder** : It processes the input sentence word-by-word using recurrent models (RNN/LSTM/GRU), produces a representation of the entire sentence in a hidden space.
- **The Decoder** : retrieves this hidden state (representation) and generates an output.

However, a notable weakness of this architecture lies in the final hidden state of the encoder, which imposes an information bottleneck. This state must encapsulate the meaning of the entire input sequence since it serves as the sole source of information for the decoder during output generation. This limitation becomes particularly pronounced when dealing with longer input sequences.

The following slide provide an overview of the inner workings of the architecture!
   ![Figure 1](visuals/1665309050800.jpeg)

1. **Word Embedding**: Each word is represented by a dense, low-dimensional vector.
2. **The Encoder**: It processes the input sentence sequentially using the RNN cells and produces the final hidden state, which is the sentence representation.
3. **The Decoder**: At each step, using the hidden state and the embedding of the previous token, it generates the next most probable token.




## The Attention Mechanism

The core concept behind the **Attention mechanism** is the assignment of weights to data, specifically to tokens (words in our context). Instead of producing a single hidden state for the entire input sequence, the encoder generates a hidden state at each step. This allows the decoder to access these individual states and make decisions regarding which portions of the input are most important and relevent at each step of the output generation.🎯 Through this mechanism, the model becomes capable of learning non trivial alignments between words in the generated translation and those present in the source sentence.

### An Illustrative Example

To grasp the significance of the Attention mechanism, consider the following example. The figure below demonstrates the alignment of the words "zone" and "Area" using an attention-based decoder, even when these words are ordered differently in the source sentence and the translation.

![Figure 2](visuals/attention_sentence.png)


The following figure explains the processe of applying the attention mechanisme on a RNN.

![Figure 2](visuals/attention_mechanisme_inRNN.png)

But Although adding the attention mechanism to RNNs has improved performance and produced much better translations,😕 training RNNs on large datasets is time-consuming as they are not parallelizable (Curse of recurrence!!).

And this is where the **the Transformeres** shine with the introduction of the **self attention**, a mechanism that opetates on all states in the same layer of a neural network.The outputs of the self-attention mechanisms serve as input to feed-forward networks.
This architecture trains much faster than recurrent models and improve the performance.

The following example illustrates the functionality of this mechanism:

![Figure 4 :](<Screenshot from 2023-09-16 16-33-30.png>)

## Transfer Learning in NLP

If you are familiar with computer vision, you have probably heard of a concept called Transfer Learning, unless you are training your CNN architectures from scratch, in which case you should consult a therapist👨‍⚕️ !

![Figure 5 :](<visuals/translearning1.png>)

🎯 Transfer learning is the process of applying an existing trained model to a new, but related task. Architecturally, this involves dividing the model into a body and a head,The head is the task-specific portion of the network, while
The body contains broad features from the source domain learned during training.During learning, the body weights learn general features, for images it learns basic features such as lines, edges, and colors... then these weights are used to initialize a new model for the new task.

![Figure 6 :](<visuals/translearning_analogie.png>)


While achieving great success in computer vision, the pre-training process for Natural Language Processing (NLP) was far from straightforward 😕. However, a significant breakthrough came with the introduction of ULMFiT, which provided the pivotal missing piece to ignite the transformer revolution.

The ULMFiT framework comprises three fundamental steps:

1. **Pre-training (Language Modeling)**: Initially, a language model is trained to predict the next word based on the preceding words within a large-scale generic corpus, typically sourced from Wikipedia text.

2. **Domain Adaptation**: The language model is subsequently fine-tuned to predict the next word, this time aligning its predictions with the target in-domain corpus.

3. **Fine-Tuning**: The language model undergoes further refinement, incorporating a classification layer for the specific target task, such as text classification.

The following exemple explain the process of building a twitter sentiment cassifier using transfer learning.

![Figure 6 :](<visuals/Screenshot 2023-10-31 at 10-19-37 1667131605803.png>)


In 2018, a monumental breakthrough occurred with the introduction of two transformer architectures: GPT and BERT. GPT only uses the decoder part of the Transformer architecture and the language modeling approach as ULMFiT. In contrast,BERT uses the encoder part of the Transformer architecture and a form of language modeling called masked language modeling 🎭.(Masked language modeling requires the model to fill in randomly missing words in a text.)

 

The collective impact of GPT and BERT was groundbreaking. They set a new gold standard across a diverse array of NLP benchmarks, marking the inauguration of a transformative chapter in the history of transformers. 🌟


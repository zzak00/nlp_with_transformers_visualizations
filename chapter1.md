# Language and Natural Language Processing

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

The core concept behind the **Attention mechanism** is the assignment of weights to data, specifically to tokens (words in our context). Instead of producing a single hidden state for the entire input sequence, the encoder generates a hidden state at each step. This allows the decoder to access these individual states and make decisions regarding which portions of the input are most important and relevent at each step of the output generation. Through this mechanism, the model becomes capable of learning intricate alignments between words in the generated translation and those present in the source sentence.

### An Illustrative Example

To grasp the significance of the Attention mechanism, consider the following example. The figure below demonstrates the alignment of the words "zone" and "Area" using an attention-based decoder, even when these words are ordered differently in the source sentence and the translation.

![Figure 2](visuals/attention_sentence.png)



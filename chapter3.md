
# NLP with Transformers chapter 3: Transformer anatomy
In this chapter, we will dive deeper into the Transformers architecture, exploring the main building blocks of a transformer model. We will first focus on constructing the attention mechanism and then integrate all the necessary components to make the encoder function. Additionally, we'll highlight the key distinctions between the encoder and decoder modules.  
tighten your seatbelt, it's time to explore the wonders of NLP✨.

## The Transformer Architecture
The original form of transformer was initially based on the encoder-decoder architecture primarily used for translation tasks,However, this design faced challenges in effectively handling long sequences. This is where the attention mechanism comes into play.  
The transformer consists of two main components :  
  
  ***Encoder :***  
  convert an input sequence into a sequence of embeddings (hidden state).
 
    
***Decoder :***  
uses its output and the encoder's hidden state to iteratively generate the output sequence.

![Figure 1](visuals/chap3visuals/encoder-decoder-linkdin.png)

Scary, isn't it🫣? Don't worry. We're here to make it simpler.

## The Encoder 
The transformer's encoder composed of  many layers. In each layer, a sequence of embeddings is passed through two main components:    
1) A multihead self attention layer
2) A fully connected feed-forward layer that is applied to each input embedding  

At the end of our encoder, we have output embeddings that maintain the same size as the inputs. They become more contextually aware. For exemple, if we refer to an "Apple phone",the word "Apple" will be updated to be more "campany like" and less " fruit like".

![figure 2](visuals/chap3visuals/apple.png)
To gain a clear understanding of how it truly works, let's begin with the most important component: the self-attention layer.
### Self-attention
As we saw earlier, each token is individually represented by a vector of either 768 or 512 dimensions. The main idea behind self-attention is to use the entire sequence to compute a weighted average matrix that describes the relationships between each embedding and the other embeddings within the same sentence. As a result, we end up with embeddings that capture context more effectively.  
To do so, we use a technique called:
#### the scaled dot product: 
There are four main steps to implement this mechanism :  

**1 )&nbsp;** Project each token into three vectors, called:
   - ***Query :&nbsp;&nbsp;*** represents the token from which the attention mechanism is getting the infotmation, it's used to compare against all the key vectors.
   - ***Key :&nbsp;&nbsp;***  tells the attention mechanism which parts of the sequence are important for understanding the query.  
   - ***Value :&nbsp;&nbsp;*** holds the informations (features) associated with each token in the sequence.
  

**2 )&nbsp;** Compute attention scores. we use the similarity fonction, which is the dot product of the embeddings matrices. Query and Keys that are similar will have a large dot product,

 **3 )&nbsp;** To prevent dealing with large numbers, we normalize the variance of the attention scores by dividing them by the square root of the dimension of the keys  $\sqrt{d_k}$ , and then we apply a softmax function to convert the column values into a probability distribution.   

**4 )&nbsp;** Multiply the attention weights by the value vectors to obtain updated embeddings.

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

#### To make it clearer, let's provide a simple example:

Let's consider the following sentence: 'I love Apple iPhone.' We will represent it in a two-dimensional embedding, where the first dimension represents the fruitiness of the word, and the second represents the technology. 


| | |
|--------------|-------------|
| **I** | [5&emsp;5] | 
| **Love** | [7&emsp;2] |
| **Apple** | [11&emsp;9] |
| **Phones** | [2&emsp;20] |


Let's now calculate the attention matrix and focus only on the word **"apple" ,** which was initially associated more with fruites than technology.


$$
\text{softmax}\left(\frac{1}{\sqrt{d_k}}*\begin{bmatrix}
5 & 5 \\
7 & 2\\
11 & 9\\
20 & 2 \\
\end{bmatrix}*\begin{bmatrix}
20 & 11 &7&5\\
2 & 9 &2 &5\\ 
\end{bmatrix}\right)
$$



we got : 
$$

\begin{array}{c|cccc}
& I & Love & Apple & Phones \\
\hline
I & * & * & * & * \\
Love & * & * & * & * \\
Apple & 0 & 0 & 0.5 & 0.5 \\
Phone & * & * & * & * \\
\end{array}
$$

We can see that the word **'apple'** is more focused on the word **'phone'** compared to the other words. Finally, let's multiply our weighted matrix by the value matrix.

$$

\begin{bmatrix}
* & * & * & * \\
* & * & * & * \\
0 & 0 & 0.5 & 0.5 \\
* & * & * & * \\
\end{bmatrix}*
\begin{bmatrix}
5 & 5 \\
7 & 2\\
11 & 9\\
20 & 2 \\
\end{bmatrix}=\begin{bmatrix}
* & * \\
* & *\\
8.5 & 14.5\\
* & * \\
\end{bmatrix}
$$

**The Updated Apple Embedding :**&emsp; [Apple] = [8.5&emsp;14.5]

We can clearly see how the embedding of the word 'Apple' becomes more technology like and less fruit like.


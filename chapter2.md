<!-- chap2 Structure : 
Intoduction:Overview of the pipline 
1-A first look at the hagging face dataset:
    -importing the emotions dataset
    -about the dataset object
    - How long are our tweets ?
2- Tokenisation :
    -character/world tokenisation (A quick explanation and why we wont use them)
    -Subworld tokenisation (including code and exemples)
    -tokenising the hall dataset (about the map() methode)
3- Training a Text classifier : 
    -Transformers as feature extractures : 
        -importing distilBert as a pretrained model
        -Extracting the last hidden states for the hall dataset
        -Creating a feature matrix
        -Visualizing the training set
        -Adding a Logisticregression Layer and validating his performance with the confusion matrix
    -Fine tuning Transformers:
        -About fine tuning
        -Loading the pretrained Distilbert for classification
        -defining the metrics : F1_score and accuracy
        -Training the model : 
            -defining the training arguments
            -training the model and calculating the metrics
            -ploting the confusion matrix
        -Error analysis (if the article is already too long we migh not include this part)
4- connecting the model with an API would be nice ...
 -->

 # NLP with Transformers chapter 2: Text classification
 Text classification is one of the most common tasks in NLP; it can be used for a broad range of applications, such as tagging customer feedback into categories or routing support tickets according to their language. 🌐 Whenever your email goes to spam, or a social media platform rejects your post or deletes your comment because it's "morally inappropriate,🍉 🇵🇸" chances are very high that a text classifier is involved. 🕵️‍♂️ Another common type of text classification is sentiment analysis, which aims to identify the polarity of a given text. In this article, we will see a step-by-step guide for building our own sentiment analysis model using the transformer architecture. 🚀

 ### About the pipeline :

The goal is to build a system that automatically classifies emotions expressed in Twitter messages about a product. 🛠️ The model will take a single tweet as input and assign one of the possible labels, including anger, fear, joy, love, sadness, and surprise.

we’ll tackle this task using a variant of BERT called ***DistilBERT***. The main advantage of this model is that it achieves comparable performance to BERT, while being significantly smaller and more efficient. And We will follow the typical pipeline for training transformer models in the **hugging face**  ecosystem.🤗

   ![Figure 1](visuals/chap2visuals/thepipline.png)


    
First, we'll load and process the dataset using the ***dataset*** library. Next, we'll tokenize the dataset using the ***Tokenizers*** library. This enables us to train our text classifier using two distinct approaches: first, utilizing transformers as feature extractors, and second, fine-tuning the DistilBERT model for a classification task. Finally, we'll load the fine-tuned model into the ***Datasets*** library for future reuse.🚀❗️


## The dataset: 

To build our emotion detector we’ll use a great dataset from an article that explored how emotions are represented in English Twitter messages and classify theme into six diffrent polarities, anger, disgust, fear, joy, sadness, and surprise. 

### Loading the dataset:

```
<!-- loading the packages -->
!pip install transformers
!pip install sentencepiece
!pip install datasets
```

```python
from datasets import load_dataset

"""NOTE: The hagging face dataset has mor than 1700 datasets 
 You can load any of theme by passing there name to the load_dataset function """
 

emotions = load_dataset("emotion")
print(emotions)
#acceding the train split
train_ds = emotions["train"]
print(train_ds)
len(train_ds)

print(train_ds.column_names)
#acceding to the first element of the train split
print(train_ds[:5])

"""  {'text': 
    ['i didnt feel humiliated',
     'i can go from feeling so hopeless to so damned hopeful just from being around someone who cares and is awake',
    'im grabbing a minute to post i feel greedy wrong',
    'i am ever feeling nostalgic about the fireplace i will know that it is still on the property', 
    'i am feeling grouchy'],
     'label': [0, 0, 3, 2, 3]} """

```
If we look inside our emotions object, we see it is similar to a Python dictionary, with each key corresponding to a different split. And we can use the usual dictionary syntax to access an individual split

```
    DatasetDict({

        train: Dataset({
            features: ['text', 'label'],
            num_rows: 16000
    }),
        validation: Dataset({
            features: ['text', 'label'],
            num_rows: 2000
    }),
        test: Dataset({
            features: ['text', 'label'],
            num_rows: 2000
    })
})

    Dataset({
        features: ['text', 'label'],
        num_rows: 16000
    })
     
    
```

### Processing the dataset :
For an access to great data visualisation APIs, it is convenint to convert a dataset object to a **Pandas** dataframe using the ***set_format()***  methode :
```python
emotions.set_format(type="pandas")
df = emotions["train"][:]
#As labels are represented as integers, 
# we are creating a new column in the dataframe for the corresponding names
def label_int2str(row):
    return emotions["train"].features["label"].int2str(row)
df["label_name"] = df["label"].apply(label_int2str)
df.head()
```

| text                                           | label | label_name |
| ---------------------------------------------- | ----- | ---------- |
| i didnt feel humiliated                         | 0     | sadness    |
| i can go from feeling so hopeless to so damned... | 0     | sadness    |
| im grabbing a minute to post i feel greedy wrong | 3     | anger      |
| i am ever feeling nostalgic about the fireplac... | 2     | love       |
| i am feeling grouchy                            | 3     | anger      |

### How long are our Tweets ?
Transformer models have a maximum input sequence length that is referred to as the
maximum context size.For DISTIlbert the maximum context size is 512 tokens,which amounts to a few paragraphs of text.

```python
#Ploting the lenth of world for every label :
df["Words Per Tweet"] = df["text"].str.split().apply(len)
df.boxplot("Words Per Tweet",by="label_name",
    grid=False,
    showfliers=False, color="black")
plt.suptitle("")
plt.xlabel("")
plt.show()

```
![Figure 1](visuals/chap2visuals/worldlenth.png)

Most tweets are between 15 and 20 words long, with a max length of around 50 words,which is well below DistilBERT’s maximum context size.Texts that are longer than a model’s context size need to be truncated, which can lead to a loss in performance if the truncated text contains crucial information.

## From Text to Tokens: Tokenization
Transformer models like DistilBERT cannot receive raw strings as input; instead, they
assume the text has been tokenized and encoded as numerical vectors. Tokenization is
the step of breaking down a string into the atomic units used in the model.The three main tokenization strategies are character tokenization, word tokenization, and subword tokenization.

### Character Tokenization :
The simplest tokenization scheme is to feed each character individually to the model. So to tokenize the followin exemple **"Tokenizing text is a core task of NLP."** we can just

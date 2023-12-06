# What you will learn?

- Generative AI?
- Large Language Models (LLMs)
- OpenAI 
- Langchain
- Vector Database
- Llama Index
- Open Source LLM model
- End to End Project

## What is Generative AI?
Generative AI generate new data based on training sample.Generative model can generate Image,Text, Audio, Videos etc. data as output.

So generative AI is a very huge topics, 
- Generative Image model
- Generative Language model

1. **Image to Image**:
   - **Definition**: Image-to-image tasks involve transforming an input image into an output image using machine learning techniques. It encompasses various tasks like image translation, image super-resolution, style transfer, and more.
   - **Examples**:
     - **Image Translation**: Converting images from one domain to another (e.g., turning summer scenes into winter scenes).
     - **Image Super-Resolution**: Enhancing the quality of an image by increasing its resolution.
     - **Style Transfer**: Applying the artistic style of one image to another while preserving the content.
   - **Approaches**: These tasks are often achieved using deep neural networks like convolutional neural networks (CNNs), generative adversarial networks (GANs), or autoencoders.

2. **Text to Image**:
   - **Definition**: Text-to-image synthesis involves generating images based on textual descriptions. The goal is to create visual representations that correspond to the provided text.
   - **Examples**:
     - **Conditional Image Generation**: Creating images based on textual input descriptions.
     - **Scene Generation**: Generating scenes or objects described in text (e.g., "a red apple on a table").
   - **Approaches**: Models for text-to-image synthesis often use architectures that combine natural language processing (NLP) and computer vision techniques, such as transformer models or architectures with attention mechanisms.

3. **Image to Text**:
   - **Definition**: Image-to-text tasks involve extracting meaningful textual information from images. This process aims to understand and describe the content of images in textual form.
   - **Examples**:
     - **Image Captioning**: Generating textual descriptions that describe the content of an image.
     - **Object Recognition**: Identifying and labeling objects or entities present in an image.
   - **Approaches**: Convolutional neural networks (CNNs) are commonly used for tasks like object detection, image classification, and image captioning. These models are trained to understand and extract information from images to produce relevant textual descriptions.

4. **Image to Image** (This seems to be a repetition; possibly meant Text to Image or another category.):
   - Image-to-image tasks, as described earlier, involve transforming an input image into an output image through various techniques such as translation, super-resolution, or style transfer.

## Where Generative AI Exists.
- Machine Learning is the subset of Artificial Intelligence
- Deep Learning is the subset of Machine Learning
- Generative AI is the subset of Deep Learning 


In the context of mappings in Language Model Mapping (LLM) or more generally in data mapping scenarios, there are four common types of relationships between data elements:

1. **One-to-One (1:1)**:
   - **Definition**: A one-to-one relationship exists when each element in one set is related to exactly one element in another set, and vice versa.
   - **Example**: In a simple translation scenario, where each word in one language corresponds directly to a single word in another language without ambiguity.

2. **One-to-Many (1:M)**:
   - **Definition**: A one-to-many relationship occurs when each element in one set is related to one or more elements in another set, but each element in the second set is related to only one element in the first set.
   - **Example**: When a word in one language has multiple possible translations in another language. For instance, the English word "run" could translate to "correr" or "corriendo" in Spanish.

3. **Many-to-One (M:1)**:
   - **Definition**: A many-to-one relationship happens when multiple elements in one set are related to a single element in another set.
   - **Example**: Multiple words in one language map to a single word in another language. For instance, words like "am", "are", and "is" in English all translate to the word "est" in French.

4. **Many-to-Many (M:M)**:
   - **Definition**: A many-to-many relationship exists when multiple elements in one set can be related to multiple elements in another set.
   - **Example**: In more complex translation or mapping scenarios where multiple words or phrases in one language may have multiple corresponding translations or meanings in another language.

# Research Paper :-  Sequence to Sequence Learning with Neural Networks
https://proceedings.neurips.cc/paper/2014/file/a14ac55a4f27472c5d894ec1c3c743d2-Paper.pdf

Tensorflow :- https://www.tensorflow.org/text/tutorials/nmt_with_attention

Attention Is All You Need :- https://arxiv.org/abs/1706.03762

## Discriminative vs Generative Model

| Aspect                | Discriminative Model                          | Generative Model                              |
|-----------------------|----------------------------------------------|-----------------------------------------------|
| Goal                  | Predict the probability of a label given input data. | Generate new data based on the learned probability distributions. |
| Focus                 | Captures the boundary between classes.        | Models the joint probability distribution of data and labels. |
| Application           | Classification tasks (e.g., logistic regression, SVM). | Generative Adversarial Networks (GANs), Hidden Markov Models (HMMs), Naive Bayes, etc. |
| Output                | Conditional probabilities (P(label | data)). | Joint probabilities (P(data, label)).        |
| Training Approach     | Tends to be simpler and focuses on learning the decision boundary. | Usually more complex as it involves modeling the full data distribution. |
| Usage                 | Often used in scenarios where the main task is classification. | Useful for tasks involving generating new samples or dealing with missing data. |
| Example               | Logistic Regression, Support Vector Machines (SVMs). | GANs for generating realistic images or text. |
| Evaluation            | Accuracy, Precision, Recall, F1-score, ROC curves, etc. | Log-likelihood, Perplexity, Reconstruction quality, etc. |

Generative AI is a subset of deep learning and Generative models are trained on huge amount of data. While training the generative model we don’t need to provide a label data, It is not possible when we have a huge amount of data, So, it's just try to see the relationship between the distribution of the data. In Generative AI we give unstructured data to the LLM model for training purpose.

# What is LLMs?
A large Language model is a trained deep learning model that understands and generate text in a human like fashion.
LLMs are good at Understanding and generating human language

### Why we call it Large Language Model?
Because of the size and complexity of the Neural Network as well as the size of the dataset that it was trained on.

Researchers started to make these models large and trained on huge datasets
That they started showing impressive results like understanding complex Natural Language and generating language more eloquently than ever.


### What makes LLM so Powerful?
In case of LLM, one model can be used for a whole variety of tasks like:-
Text generation, Chatbot, summarizer, translation, code generation 
& so on …

So, LLM is subset of Deep Learning & it has some properties merge with
Generative AI

### Few milestone in large language model
BERT: Bidirectional Encoder Representations from Transformers (BERT) was developed by Google

GPT: GPT stands for "Generative Pre-trained Transformer".The model was developed by OpenAI 

XLM: Cross-lingual Language Model Pretraining by Guillaume Lample, Alexis Conneau.

T5: The Text-to-Text Transfer Transformer It was created by Google AI

Megatron: Megatron is a large, powerful transformer developed by the Applied Deep Learning Research team at NVIDIA

M2M-100: multilingual encoder-decoder (seq-to-seq) model researchers at Facebook

## Open Source Models
- BLOOM
- Llama 2
- PaLM 
- Falcon 
- Claude
- MPT-30B
- Stablelm
So on ….

## What can LLMs be used for?
- Text Classification
- Text Generation
- Text Summarization
- Conversation AI like chatbot, Question Answering
- Speech recognition and Speech identification
- Spelling Corrector
So on……


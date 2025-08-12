# Fine Tuning
----------------

### 1. Text Classification: Fine Tuning BERT With HuggingFace
------------------
In this notebook, I will walk through the complete process of fine-tuning a [BERT (Bidirectional Encoder Representations from Transformers)](https://en.wikipedia.org/wiki/BERT_(language_model)) model using the [HuggingFace ecosystem](https://huggingface.co/). BERT has become a cornerstone of modern NLP due to its ability to capture bidirectional context and deliver strong performance across a wide range of language understanding tasks such as classification, named entity resolution and question answering. In this post I will build of [prior posts on text classification](https://michael-harmon.com/blog/NLP4.html) by fine tuning a BERT model to classify the topic of papers in [arxiv](arxiv.org) by their abstract text. By the end of this post, I will have a working, fine-tuned BERT model ready for inference on the [Hugging Face Model Hub](https://huggingface.co/models).

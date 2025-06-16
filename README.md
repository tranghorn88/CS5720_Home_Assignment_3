# CS5720_Home_Assignment_3
**Student Name:** Trang Horn

**Student ID:** 700683454

# Question 1: RNN for Character-Level Text Generation 

## Description:

This code implements a character-level Recurrent Neural Network (RNN), using LSTM layers to generate English text. The model is trained on Shakespeare’s text and learns to predict the next character in a sequence. It then uses the trained model to generate new, original text character-by-character. Specifically, it performs the following tasks:
1. Load and preprocess a text dataset.
2. Convert the characters into numerical representations.
3. Build an RNN with TensorFlow using LSTM layers.
4. Train the model to predict the next character in a sequence.
5. Generate text using temperature scaling to control randomness.

## Discussion:
Temperature scaling controls the randomness of predictions in text generation by adjusting the distribution of probabilities output by the model’s softmax layer. With low temperature (e.g., 0.2), the model chooses high-probability characters more confidently. Therefore, it results in less creative, more repetitive, but grammatically safer text. Meanwhile, with high temperature (e.g., 1.2), the model is more likely to sample low-probability characters and results in more creative, diverse, but possibly less coherent text.

# Question 2: NLP Preprocessing Pipeline

## Description
The code implements a basic NLP preprocessing function using the NLTK library in Python. It demonstrates the foundational steps commonly used in text processing before feeding data into NLP models.
Input Sentence Used:
"NLP techniques are used in virtual assistants like Alexa and Siri."
The following preprocessing steps was performed on the input sentence:
1. Tokenization – Splits the input sentence into individual words (ignores punctuation).
2. Stopword Removal – Removes common English stopwords like “the”, “are”, “in”.
3. Stemming – Reduces each remaining word to its root form using the Porter stemmer.

## Short Answer Questions
1. Stemming and lemmatization are both techniques used in NLP to reduce words to their base or root form. But they work differently. Stemming reduces a word to its base by using crude, rule-based heuristics, chopping off word endings, sometimes incorrectly. It is less accurate. For example: "running" → run (correct), "happily" → "happili" (inaccurate).
Lemmatization returns the dictionary form (lemma) of a word using vocabulary and POS tags. Wile with stemming the output may not be a real word, lemmatization always outputs a valid base word and it is more accurate. For example: "running" → run (correct), "happily" → "happily" (correct lemma).

2. Why might removing stop words be useful in some NLP tasks, and when might it actually be harmful?
Stop words are common words in a language that are often filtered out during natural language processing (NLP) tasks because they do not carry significant meaning on their own. In tasks like document classification or topic modeling, stopwords add little meaning and can be removed to reduce noise, speed up processing and improve model performance. However, stop words are not always useless. In tasks like sentiment analysis or question answering, stopwords such as "not", "is", or "was" may carry essential meaning. Removing them could distort the sentence’s intent.

# Question 3: Named Entity Recognition with SpaCy 

## Description
This code demonstrates how to perform Named Entity Recognition (NER) using the `spaCy` NLP library. The goal is to extract real-world entities from a given sentence and provide:
- The entity text
- The label (e.g., PERSON, GPE, DATE)
- The start and end character positions in the sentence
Input Sentence Used:
"Barack Obama served as the 44th President of the United States and won the Nobel Peace Prize in 2009."

## Short Answer Questions
1. How does NER differ from POS tagging in NLP?
NER or Named Entity Recognition identifies real-world objects such as people, places, dates, or organizations within text. For example: "Barack Obama" → PERSON, "May 17th, 2009" → DATE. 
Meanwhile, POS or Part-of-Speech tagging assigns grammatical roles to words, such as noun, verb, adjective. For example: "Obama" → NNP (proper noun), "won" → VBD (verb, past tense). 
POS tagging is syntactic; NER is semantic.

2. Financial institutions such as banks, hedge funds, and traders use NER to extract entities such as company names, stock tickers, currencies, dates, and monetary values, etc. from breaking news articles. In these cases, NER is applied to real-time news feeds to identify: Organizations (e.g., "Apple Inc."), Monetary values( e.g., "$3.2 billion"), Dates( e.g., "May 17th, 2024"), Locations( e.g., "New York Stock Exchange"). This structured information is then fed into automated trading algorithms or risk assessment dashboards.
   
  Google, Alexa, and Siri use NER to understand user queries more effectively by identifying key people, places, dates, and products. For example, NER helps isolate the core entities in natural language queries:
  "When was Barack Obama born?" → PERSON  
  "What's the weather in Paris?" → GPE (geo-political entity)  
  The search engine then links those entities to structured databases (like Wikidata or Google Knowledge Graph) to return accurate, relevant answers.
  Another example will be when issuing the query: “Schedule a meeting with Dr. I Hua at the University of Central Missouri on May 5th, 2025”  
  NER Result will be:  
  Dr. I Hua → PERSON  
  University of Central Missouri → ORGANIZATION  
  May 5th, 2025 → DATE  
  This information is used to populate calendar fields automatically.

# Question 4: Scaled Dot-Product Attention 

## Description
This code implements the scaled dot-product attention mechanism used in Transformer models. It is a fundamental operation that allows the model to focus on different parts of the input sequence when producing an output, and it is a key component of the attention mechanism in models like BERT and GPT.
The attention mechanism performs the following steps:
1. Compute the dot product of **Q** and **Kᵀ**.
2. Scale the scores by dividing by the square root of the key dimension (√d).
3. Apply the softmax function to normalize the scores into attention weights.
4. Multiply the attention weights by **V** to get the final output.

## Short Answer Questions
1. Why do we divide the attention score by √d in the scaled dot-product attention formula?
Dividing by the square root of the key dimension (√d) helps to prevent extremely large dot product values, which could lead to very small gradients after the softmax function.  Also, scaling keeps the variance of the dot products stable, helping the softmax function produce more balanced and learnable probabilities.

2. How does self-attention help the model understand relationships between words in a sentence?
Self-attention allows the model to weigh the importance of each word in a sentence relative to every other word, enabling the model to understand contextual relationships (e.g., "bank" of a river vs. "bank" for money), capture long-range dependencies, even between distant words in the input, and dynamically focus attention on relevant words during translation, summarization, or other sequence-based tasks.

# Question 5: Sentiment Analysis using HuggingFace Transformers

## Description
This code demonstrates how to use the HuggingFace `transformers` library to perform sentiment analysis with a pre-trained model. The model classifies the sentiment of a given sentence as either POSITIVE or NEGATIVE and returns a confidence score. Specifically:
- Load a pre-trained sentiment analysis pipeline using HuggingFace.
- Analyze the following input sentence: 
  "Despite the high price, the performance of the new MacBook is outstanding."

## Short Answer Questions
1. BERT is built using an encoder-only architecture, which is optimized for understanding tasks such as text classification, named entity recognition, and question answering, etc. Meanwhile, GPT uses a decoder-only architecture, which is designed for generating text, making it suitable for tasks like text completion, dialogue systems, and story generation, etc.
2. Using pre-trained models (like BERT or GPT) is beneficial for NLP applications instead of training from scratch due to the following reasons:
 Pre-trained models are trained on massive datasets. They generalize well across tasks like classification, translation, and summarization and can be fine-tuned with much less data. Therefore, it helps saves time and compute. In addition, they capture deep contextual and syntactic relationships, providing better performance.


TODO: train the model (100 hours... CUDA?)

# movie-classifier

## Challenge Goal

Develop a predictive model that analyzes movie synopses and assigns appropriate genres. The challenge is to achieve high prediction accuracy and ensure the quality and efficiency of your code.

## Evaluation Criteria

Machine Learning Model Performance: Mean Average Precision at K for the top 5 predicted genres. Note that this metric keeps the ranking into account.

## Dataset

The dataset includes movie synopses with corresponding genres, split into training and test sets. The training set is for model development, and the test set evaluates your model's performance.
Submission Guidelines

Submit predictions in a specified format

Good luck!

# (Multi Label) Text classification

Text classification is a common NLP task that assigns a label or class to text. One of the most popular forms of text classification is sentiment analysis, which assigns a label like 🙂 positive, 🙁 negative, or 😐 neutral to a sequence of text.

Classification problems can be binary, multi-class or multi-label. In a binary classification problem, the target label has only two possible values. The difference between binary and multi-class classification is that multi-class classification has more than two class labels. A multi-label classification problem has more than two class labels, and the instances may belong to more than one class.

https://huggingface.co/docs/transformers/en/tasks/sequence_classification

Binary text classification with transfer learning (DistilBERT)

https://medium.com/analytics-vidhya/an-introduction-to-multi-label-text-classification-b1bcb7c7364c

Multi-label text classification with problem transformation (logistic regression)

We have to remove all the stopwords and special characters in the articles. We have also used a snowball stemmer. A stemmer transforms all the different forms of a word into a single word.

- Binary relevance transforms a multi-label classification problem with L labels into L separate single-label binary classification problems. In this approach, each classifier predicts the membership of one class. The union of the predictions of all the classifiers is taken as the multi-label output. As this is an easy approach, binary relevance is very popular. But, the main drawback of binary relevance is that it ignores the possible correlations between the classes.

- Classifier chains is similar to binary relevance. But it takes label correlation into account. This approach uses a chain of classifiers where each classifier uses the predictions of all the previous classifiers as input. The total number of classifiers is equal to the number of classes. As this method uses label correlation, it gives better results than binary relevance.

- The idea behind label powerset is to transform the multi-label classification into a multi-class problem. In this approach, a classifier is trained on all the unique label combinations in the training dataset. As the number of class labels increases, the number of unique label combinations also increase. This would make it expensive to use this approach. Another disadvantage is that it predicts only the label combinations that are seen in the training dataset.

We have discussed the problem transformation method to perform multi-label text classification. There are several approaches to deal with a multi-label classification model. 

We also noticed that the data is imbalanced. We can further improve our model by applying techniques like MLSMOTE to balance the input data. It is also a good idea to use deep learning techniques like LSTM.

https://huggingface.co/blog/Valerii-Knowledgator/multi-label-classification

Multi-label text classification with transfomers (DeBERTa)

We tokenise the dataset (AutoTokenizer) and process labels for multi-label classification. It's more efficient to dynamically pad the sentences to the longest length in a batch during collation instead of padding the whole dataset to the maximum length (DataCollatorWithPadding).

In this article, we have demonstrated how to build a multi-label text classifier from scratch using Transformers and Datasets. We leveraged the DeBERTa encoder model and fine-tuned it on a custom biotech news dataset with 31 labels. After tokenising the text and encoding the multi-hot labels, we set up the training loop with helpful metrics for monitoring. The model can now accurately classify biotech news articles into multiple relevant categories. This end-to-end workflow showcases how new datasets can be prepared and fed into powerful pre-trained models like DeBERTa to create custom and powerful text classifiers for real-world applications. Now, you can add more data and labels to train the model tailored to your specific requirements and reach your professional goals. And we're happy to help with your research & development efforts!

https://keras.io/examples/nlp/multi_label_classification/

https://paperswithcode.com/task/multi-label-text-classification

# Git Large File Storage

source: https://git-lfs.com/

In each Git repository where you want to use Git LFS, select the file types you'd like Git LFS to manage (or directly edit your .gitattributes).

```
git lfs track "*.psd"
```

Now make sure .gitattributes is tracked:

```
git add .gitattributes
```

There is no step three. Just commit and push to GitHub as you normally would;


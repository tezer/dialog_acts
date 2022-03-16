# Automatic identification of action-items

## An outline of the technologies I would need to bring this project to life:
I am planning to use transformers for this project, so I will need to use the following technologies:
- pytorch
- huggingface
- transformers
- numpy
- pandas
- matplotlib
- sklearn
- wandb (for learning tracking)

For a more efficient production I will need 
- ONNX Runtime, a cross-platform inference and training machine-learning accelerator
- NVIDIA Triton Inference Server, for GPU use optimization 

## An outline of deployment and building of this project.
- Research and development (2-4 weeks)
    - Look for papers describing possible solutions to this problem
    - Look for existing solutions with code
    - Select 2-3 most promising approaches
    - Find a suitable data set or develop one
    - Research and develop a solution based on the top performing approach
- Deployment (1-2 weeks)
    - Convert the model to ONNX format
    - Deploy the model on Triton
- Feedback and testing (1-2 months)
    - Monitor the performance of the model
    - Collect feedback from users
    - Fix errors and bugs

## Basic prototype of the action items detection code

### Approach
The prototype is based on the following paper:
[**Sheshadri, S. (2019). Identifying Action related Dialogue Acts in Meetings**](https://www.diva-portal.org/smash/get/diva2:1380622/FULLTEXT01.pdf)

The author of the paper found that the most effective way to detect action-items is to use binary classification based on fine-tuned BERT-based models.
I implemented this approach for two models: **bert_base_uncased** and **bert_large_uncased**.

### Data
I used the freely available dataset [**SILICONE**](https://huggingface.co/datasets/silicone) described in [**Hierarchical Pre-training for Sequence Labelling in Spoken Dialog**](https://www.aclweb.org/anthology/2020.findings-emnlp.239).
The dataset is a collection of resources for training, evaluating, and analyzing natural language understanding systems specifically designed for spoken language. All datasets are in the English language and cover a variety of domains including daily life, scripted scenarios, joint task completion, phone call conversations, and television dialogue.
I used corpus **dyda_da** (dayly dialogue acts) and used _commissive_ and _directive_ utterances as actionable items.
Unfortunately, the aforementioned types are not equivalent to the actionable items in the context of business minutes. But they are the closest match that I was able to find.

### Algorithm
For model fine-tuning, I used 2 dense layers for _bert_base_uncased_ and 3 layers for _bert_large_uncased_. AdamW for optimization and loss function is CrossEntropyLoss.
The learning rate was 1e-3 with BERT layers frozen.

## Results
After training for 10 epochs, I tested the model on the test set and found that both models achieved the same accuracy of 0.7
One of the ways of improvement of the model is to train the model with the BERT layers unfrozen and train it for more epochs.

The sentences extracted from the supplied files seem to have too many false positives.

The sentences from the assignment page are processed quite well

### Project structure
There are four files:
- **da_model.py**: contains the model and basic settings
- **train.py**: contains the training and evaluation functions
- **predict.py**: contains the prediction function and test set evaluation
- **test.py**: contains three options for testing the model

The checkpoints are saved in the *checkpoints* folder.

The *data* folder contains the data files. 

### How to run test.py
- if the test.py has a filename as its argument, it will try to load a list of sentences from the file specified by the argument.
- if the test.py has no argument, it will run the list of sentences from the task specification and turn into an interactive mode, where a user can type in any sentence to predict if it contains an actionable item.
- to change model type, edit _model_name_ in _da_model.py_ (line 2) accordingly


### TODO
- Train the model with unfrozen BERT layers at a learning rate of 1e-5
- Try other transformer architecture types
- Use Wandb to monitor the training and testing process

### Alternative approach
The alternative approach is based on findings described in some papers that the structure of the dialog is very important for accurate detection of the dialog acts.
For example, [**Dialogue Act Classification with Context-Aware Self-Attention**](https://arxiv.org/abs/1904.02594) describes one. The authors treated the task as a sequence labeling problem using hierarchical deep neural networks, leveraging the effectiveness of a context-aware self-attention mechanism coupled with a hierarchical recurrent neural network.

I used an existing implementation of the model, available on GitHub [here](https://github.com/PolKul/CASA-Dialogue-Act-Classifier).

Unfortunately, this model is too slow to train and there is no publicly available checkpoints of the model. I was able to run only one epoch on my hardware and the results were not good for this point.

If more powerful hardware were available, I would like to explore this approach and see if it could be used to detect actionable items in the dialog.

## How would you measure whether or not this feature is doing what its intending to do?
First, I would like to know how well the model is performing on the test set. To measure this, I use per class Precision, Recall, and F1-score.

Secondly, looking into actual classification results may help to understand the model and find way of improving it. It may be data preprocessing, model architecture, or hyperparameters. Adding some heuristics to the pipeline may help to improve the results by eliminating some repetitive cases.

For model settings and feature tuning, I usually use WandB.

## If we are to put this in front of users today, what are the pros and the cons? Should we put it in front of customers?
Pros:
- It is capable of finding actionable items in the dialog (separate sentences)
- It is reasonably fast

Cons:
- It is not tested extensively on the relevant data
- It may be omitting some important cases and having too many false positives that would annoy users

Bottom line: I would not put this in front of users today.
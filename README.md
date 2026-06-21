# RNN-SMS-Spam-Detection
This project transitions you from structured data into sequence modeling, where the order and semantic context of words matter. Using a Recurrent Neural Network (RNN)—specifically an LSTM or GRU—allows your model to maintain a memory of previous words in an SMS, making it exceptionally good at catching sophisticated phishing and spam structures.
# RNN Classification: SMS Spam Detection (Discovery-to-Action Strategy)

An advanced, sequence-aware Natural Language Processing (NLP) framework that utilizes Recurrent Neural Networks (RNNs) with **Long Short-Term Memory (LSTM)** layers to classify SMS text data into spam and ham. 

This repository applies the **Discovery-to-Action (DTA) Framework**, moving from raw text sequencing metrics directly into actionable, precision-first business rules tailored for real-world user experiences (UX) in personal messaging software.

---

## Preprocessing & Network Architecture

Unlike standard tabular models, standard sequential models require raw text to be converted into static numerical shapes while preserving chronological word structures.
[ Raw Text String ] -> "URGENT! Click here..."
            |
            v
   [ Tokenizer Object ] -> Maps unique words to integers
            |
            v
   [ Numeric Sequence ] -> [41, 102, 12, 5]
            |
            v
   [ pad_sequences()  ] -> Truncates/Pads to structural size (shape: 50,)
            |
            v
   [ Embedding Layer  ] -> Projects index numbers into continuous dense space (32-dim)
            |
            v
   [ LSTM Layer (32)  ] -> Captures semantic context and time dependencies
            |
            v
   [ Dropout (0.2)    ] -> Drops random activations to prevent pattern memorization
            |
            v
   [ Sigmoid Dense    ] -> Computes explicit probability metrics (0.0 to 1.0)

### Key Engineering Parameters
* **Vocabulary Cap (`MAX_WORDS`):** 5,000 top unique words are tokenized to exclude erratic typos or noisy variations.
* **Sequence Standardization (`MAX_SEQUENCE_LENGTH`):** Standardized to 50 tokens using post-padding (`padding='post'`) and post-truncation to align text shapes for batch matrix distribution.
* **Regularization & Stabilization:** Equipped with a `Dropout(0.2)` regularization tier and configured with an `EarlyStopping` callback tracking `val_loss` with a patience limit of 3 epochs to avoid overfitting.

---

## Evaluation Rationale: Precision is King

In standard AI applications, Overall Accuracy is evaluated uniformly. However, for communication and messaging software, **Precision is the defining metric.**

$$\text{Precision} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Positives (FP)}}$$

### The Real-World Cost of Classification Errors:
* **False Negative (Spam leaks into Inbox):** Low Cost. The user experiences a minor, brief inconvenience deleting the message.
* **False Positive (Legitimate message routed to Junk):** **Catastrophic Cost.** If an automated system marks a crucial bank transaction link, emergency notification, or medical verification message as spam, the user may miss business-critical info. 
* *By prioritizing High Precision over global accuracy, we strictly minimize False Positives, ensuring vital communications are never intercepted.*

---

## Action Phase: Tiered UX Routing Thresholds

To translate the neural network's raw sigmoid output probability into an enterprise-grade user workflow, this project rejects binary 50% thresholds in favor of a **Data-Backed Tiered Confidence Strategy**:

              +--------------------------------+
              |    Model Output Probability    |
              +---------------+----------------+
                              |
        +---------------------+---------------------+
        |                     |                     |
        v                     v                     v
 [ Prob < 15% ]        [ 15% <= Prob <= 95% ]     [ Prob > 95% ]
        |                     |                     |
        v                     v                     v
 +--------------+      +--------------+      +--------------+
 | Primary      |      | Gray-Zone    |      | Immediate    |
 | Inbox        |      | Quarantine   |      | Junk Folder  |
 | (Clean)      |      | (User Review)|      | (Automated)  |
 +--------------+      +--------------+      +--------------+

1. **The Primary Inbox Pass ($< 15\%$):** Clean text streams pass straight into the main feed with native delivery alerts active.
2. **The Gray-Zone Quarantine ($15\%$ to $95\%$):** Placed in a specialized holding archive. The user receives no push sound alerts, but can easily inspect the folder to verify borderline messages manually.
3. **The Immediate Junk Route ($> 95\%$):** Definitively routed to the core spam folder without alerting the user, minimizing UI noise with high confidence.

---

## Ethical & Deployment Considerations

1. **User Privacy Boundaries:** Training models directly on user message histories requires highly secure processing pipelines. In production settings, weights should be computed using federated learning architectures or on-device filtering frameworks to prevent direct data harvesting.
2. **Dynamic Context Drift:** Phishing patterns shift constantly during global events. The model architecture requires automated shadow-evaluation triggers that flag performance degradation when text features evolve beyond original training tokens.

---

## Getting Started

### Prerequisites
```bash
pip install numpy pandas matplotlib scikit-learn tensorflow

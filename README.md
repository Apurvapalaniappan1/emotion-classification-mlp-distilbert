# Emotion Classification: Lightweight MLP vs. DistilBERT

This project compares a lightweight PyTorch classifier trained from scratch with a pretrained DistilBERT transformer on the [DAIR-AI Emotion dataset](https://huggingface.co/datasets/dair-ai/emotion). Both models use the same official train, validation, and test splits and are evaluated using accuracy and macro-F1.

The project also evaluates both models on a manually annotated set of literary sentences to examine how well they generalize beyond the social-media language used for training.

## Highlights

- 20,000 labelled English texts across six emotions
- Custom 15,000-token WordPiece vocabulary for the lightweight model
- PyTorch MLP with systematic learning-rate, hidden-size, and batch-size experiments
- Fine-tuned `distilbert-base-uncased` model
- Accuracy, macro-F1, per-class, confusion-matrix, and error analysis
- Out-of-domain evaluation on 50 manually annotated literary sentences
- Comparison of predictive performance, model size, and training time

## Dataset

The dataset contains short English texts labelled as one of six emotions:

- sadness
- joy
- love
- anger
- fear
- surprise

The official split contains 16,000 training examples, 2,000 validation examples, and 2,000 test examples. The dataset is connected to the CARER work by Saravia et al. (2018).

The dataset is downloaded through Hugging Face when the notebook is executed and is not redistributed in this repository.

## Models

### Lightweight MLP

The first model uses a WordPiece tokenizer trained only on the training texts. Each token is mapped to a learned embedding, the token embeddings are summed, and a `tanh` transformation is followed by a six-class output layer.

The selected configuration uses:

- Learning rate: `1e-3`
- Hidden size: `8`
- Batch size: `16`
- Trainable parameters: `120,054`

### DistilBERT

The second model fine-tunes `distilbert-base-uncased` using the same data splits and evaluation metrics.

The fine-tuning configuration uses:

- Learning rate: `2e-5`
- Weight decay: `0.01`
- Batch size: `16`
- Maximum epochs: `4`
- Trainable parameters: `66,958,086`
- Best-model selection by validation macro-F1

## Results

### Official test set

| Model | Accuracy | Macro-F1 | Trainable parameters | Training time |
|---|---:|---:|---:|---:|
| Custom MLP | 0.8715 | 0.8235 | 120,054 | 0.79 min |
| DistilBERT | **0.9275** | **0.8809** | 66,958,086 | 3.57 min |

DistilBERT improved accuracy by 5.60 percentage points and macro-F1 by 5.74 percentage points. The largest class-level F1 improvement was for love, which increased from 0.7089 to 0.8318. Surprise remained the most difficult class.

### Out-of-domain literary set

| Model | Accuracy | Macro-F1 | Accuracy drop | Macro-F1 drop |
|---|---:|---:|---:|---:|
| Custom MLP | 0.4000 | 0.3649 | 0.4715 | 0.4585 |
| DistilBERT | **0.5600** | **0.4745** | **0.3675** | **0.4064** |

DistilBERT generalized better to the literary sentences, but both models experienced a substantial decline compared with the official test set. This shows that transfer learning improved robustness without eliminating domain shift.

## Main findings

- Pretrained contextual representations improved both overall and class-level performance.
- DistilBERT was more robust to unfamiliar vocabulary and literary writing styles.
- The lightweight MLP remained competitive considering its much smaller size.
- DistilBERT used approximately 558 times more trainable parameters and took about 4.5 times longer to train in this experiment.
- Love and surprise were difficult because of class imbalance and semantic overlap with joy and fear.
- Strong performance on the original test distribution did not guarantee comparable performance on a different language domain.

## Repository structure

```text
emotion-classification-mlp-distilbert/
├── README.md
├── notebooks/
│   └── emotion_classification.ipynb
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Running the notebook

1. Open `notebooks/emotion_classification.ipynb` in Google Colab.
2. Select a GPU runtime for DistilBERT fine-tuning.
3. Run the notebook from top to bottom.

The notebook downloads the public dataset, pretrained tokenizer, and DistilBERT checkpoint during execution. No Hugging Face credentials are required.

To install the Python dependencies locally:

```bash
python -m pip install -r requirements.txt
```

Training times depend on the available GPU and software environment. The reported times came from the executed Colab experiment included in this repository.

## Limitations

- The literary evaluation set contains only 50 examples.
- The literary annotations were produced by one annotator.
- Its class distribution is imperfect, and its texts come from a limited set of works.
- Only one pretrained transformer checkpoint was fine-tuned.
- The two models differ substantially in pretraining, parameter count, and computational cost.

The out-of-domain results should therefore be interpreted as a focused robustness experiment rather than a general benchmark.

## References

- Saravia, E., Liu, H.-C. T., Huang, Y.-H., Wu, J., & Chen, Y.-S. (2018). [CARER: Contextualized Affect Representations for Emotion Recognition](https://aclanthology.org/D18-1404/). EMNLP 2018.
- [DAIR-AI Emotion dataset on Hugging Face](https://huggingface.co/datasets/dair-ai/emotion)
- [DAIR-AI Emotion dataset repository](https://github.com/dair-ai/emotion_dataset)
- [DistilBERT model card](https://huggingface.co/distilbert/distilbert-base-uncased)

## Licence

The project code is released under the MIT License. The external dataset and pretrained model remain subject to their respective licences and terms.


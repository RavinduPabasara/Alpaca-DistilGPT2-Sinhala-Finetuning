# Sinhala Fine-Tuned distilGPT-2 for Text Generation

## Project Overview

This project focuses on fine-tuning the `distilgpt2` model for Sinhala text generation tasks. The dataset used for fine-tuning includes instruction-based text inputs, and the model is designed to generate coherent and contextually appropriate responses in Sinhala.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Model](#model)
- [Fine-Tuning Process](#fine-tuning-process)
- [Usage](#usage)
- [Results](#results)
- [Saving and Loading the Model](#saving-and-loading-the-model)
- [Google Drive Integration](#google-drive-integration)
- [License](#license)

## Installation

To run this project, ensure you have the following packages installed:

```bash
pip install pyarrow==14.0.2
pip install requests==2.31.0
pip install transformers torch pandas
pip install datasets==2.18.0
```

These dependencies are necessary for handling the dataset, tokenizing inputs, and fine-tuning the model.

## Dataset

The dataset used in this project is stored in a JSON file named `alpaca-sinhala.json`. It includes pairs of `instruction` and `input` fields, which are combined to form the `prompt`. The corresponding `output` field is the target text that the model is trained to generate.

The dataset is converted into a Hugging Face `Dataset` object for efficient processing and batching.

## Model

This project fine-tunes the `distilgpt2` model with a tokenizer based on `xlm-roberta-base`. The tokenizer is modified to include special tokens as needed.

## Fine-Tuning Process

The fine-tuning process involves:

1. **Tokenization**: The prompts and outputs are tokenized with padding and truncation to a maximum length of 64 tokens.
2. **Training**: The model is trained using `AdamW` optimizer and a linear learning rate scheduler. The training loop includes gradient accumulation to stabilize updates, and mixed precision training is employed for efficiency.

Fine-tuning is conducted over three epochs, with loss printed at each step.

## Usage

Once the model is fine-tuned, it can be used to generate Sinhala text based on a given prompt.

```python
from transformers import GPT2LMHeadModel, XLMRobertaTokenizer
import torch

# Load the fine-tuned model and tokenizer
model_path = "./sinhala_fine_tuned_distilgpt2_2"
tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Define the prompt
prompt = "අක්ෂර වින්‍යාසය සහ ව්‍යාකරණ වැරදි සඳහා මෙම වාක්‍යය ඇගයීමට ලක් කරන්න: ඔහු තම ආහාර වේල සකසා ආපනශාලාවෙන් පිටව ගියේය"

# Tokenize the input
inputs = tokenizer(prompt, return_tensors='pt')

# Generate text
model.eval()
with torch.no_grad():
    outputs = model.generate(
        inputs['input_ids'],
        max_length=100,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id
    )

# Decode the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

## Results

During the training, the model's loss is tracked and printed for each epoch, allowing you to monitor the progress and effectiveness of the fine-tuning process.

## Saving and Loading the Model

After training, the model and tokenizer are saved in a specified directory:

```python
save_directory = "./sinhala_fine_tuned_distilgpt2_2"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
```

## Google Drive Integration

The fine-tuned model is saved to Google Drive for easy access and backup. Ensure that Google Drive is mounted in your Colab environment before saving:

```python
from google.colab import drive
import shutil

# Mount Google Drive
drive.mount('/content/drive')

# Copy the model to Google Drive
drive_path = '/content/drive/MyDrive/sinhala_fine_tuned_distilgpt2_2'
shutil.copytree('./sinhala_fine_tuned_distilgpt2_2', drive_path)
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

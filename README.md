# dermachecker-skin-lesion-classifier
Melanoma classification using CNN (transfer learning)

## Overview

Dermachecker is a deep-learning-powered image classification app built to classify skin lesion images into one of seven categories.
The model is based on DenseNet121 and fine-tuned on a dermoscopic skin image dataset.

https://github.com/user-attachments/assets/bd6800d7-f960-44eb-9eb5-c34e8209abb6

This was a personal project I worked on a few years ago to explore AI in healthcare and medical image classification. I recently rebuilt the site (I'd lost the original HTML/CSS) and fixed a deployment bug that was pointing the app at the wrong checkpoint filename, so it's a working demo again.

## Run it

```bash
pip install -r requirements.txt
python server.py
```

Then open `http://localhost:8790`. This serves the full site — landing page, About/How-To/Founder/Disclaimer sections, and a **working Test & Result panel**: drop in an image and it runs a real forward pass through the trained model and returns a prediction with a top-3 confidence breakdown.

`ml-model.py` is a Streamlit-based alternative front end for the same model, if you'd rather run `streamlit run ml-model.py`.

## Features
🧠 Deep Learning Model — DenseNet121 fine-tuned for multiclass classification

🩻 7 Skin Lesion Classes:

- Actinic keratoses
- Basal cell carcinoma
- Benign keratosis-like lesions
- Dermatofibroma
- Melanocytic nevi
- Melanoma
- Vascular lesions

🖼 Interactive App — Upload an image and get instant classification results

## Known issue (fixed in `server.py`)

The label-mapping dict used to build the training set had a copy-paste typo: `'mel': 'dermatofibroma'` instead of `'mel': 'melanoma'` (see the notebook, the cell right before `df_original` is built). Because pandas treats that lowercase `'dermatofibroma'` as a category distinct from the real, capitalized `Dermatofibroma` class, one of the model's seven output classes (index 6) was actually trained on real melanoma images under the wrong name — and the originally deployed app displayed it as "Dermatofibroma."

`server.py` corrects the displayed label for that class to **Melanoma**, which is what the training images for that class actually are. The model itself wasn't retrained — only the label shown for its 7th output node was fixed. Worth knowing if you extend the label set or retrain.

## Disclaimer

This is a personal project, not a medical device. It has not been clinically validated and should never be used to diagnose or rule out a real skin condition. If you're concerned about a lesion, see a licensed dermatologist.

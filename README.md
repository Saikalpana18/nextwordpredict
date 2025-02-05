# Next Word Predictor using RNN/LSTM

This project is a Streamlit-based web application that predicts the next word in a given sentence using a trained RNN/LSTM model.

## Demo
Check out the live demo of the project: [Next Word Predictor](https://next-word-prediction.streamlit.app)

## Features
- Predicts the next word for a given input text.
- Uses a pre-trained LSTM model trained on the IMDB dataset.
- Simple UI built with Streamlit.

## Installation
### Clone the Repository
```sh
git clone https://github.com/your-username/next-word-predictor.git
cd next-word-predictor
```

### Install Dependencies
```sh
pip install -r requirements.txt
```

## Running the Application
```sh
streamlit run app.py
```

## Files
- `app.py`: The Streamlit application.
- `next_word_model.h5`: Pre-trained LSTM model.
- `tokenizer.pkl`: Tokenizer for text preprocessing.
- `requirements.txt`: List of dependencies.

## License
This project is licensed under the MIT License.


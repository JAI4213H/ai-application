# Ink / MNIST

A small TensorFlow Keras CNN that recognizes handwritten digits through a Flask website.

## Run locally

1. Install Python 3.10 or newer.
2. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. Start the project:

   ```powershell
   .\start_server.bat
   ```

The launcher uses the included `digit_model.keras`. If the model is missing, it trains a new one from the local MNIST files in `archive/`, then starts Flask at `http://127.0.0.1:5000`.

## Files

- `model.py` trains and saves the CNN.
- `app.py` serves the website and prediction API.
- `templates/index.html` contains the drawing interface.
- `start_server.bat` checks the model and starts the server.
- `archive/` contains the MNIST IDX files used for training.
- `digit_model.keras` is the saved trained model.

To retrain the model, run `python model.py`.

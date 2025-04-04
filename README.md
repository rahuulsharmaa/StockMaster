# StockMaster

🚀 **StockMaster** is a stock analysis and prediction web application built with **FastAPI**, **Jinja2**, and **PostgreSQL**.

## 🧠 Features

- Real-time stock data visualization
- Technical indicators analysis
- LSTM-based stock prediction
- User-friendly UI with Jinja2 templates
- PostgreSQL database integration

## 📂 Project Structure

├── ./
│   .env
│   .gitignore
│   main.py
│   pythonstruture.py
│   README.md
│   requirement.txt
│   ├── .git/
│   │   COMMIT_EDITMSG
│   │   config
│   │   description
│   │   HEAD
│   │   index
│   │   ├── hooks/
│   │   │   applypatch-msg.sample
│   │   │   commit-msg.sample
│   │   │   fsmonitor-watchman.sample
│   │   │   post-update.sample
│   │   │   pre-applypatch.sample
│   │   │   pre-commit.sample
│   │   │   pre-merge-commit.sample
│   │   │   pre-push.sample
│   │   │   pre-rebase.sample
│   │   │   pre-receive.sample
│   │   │   prepare-commit-msg.sample
│   │   │   push-to-checkout.sample
│   │   │   sendemail-validate.sample
│   │   │   update.sample
│   │   ├── info/
│   │   │   exclude
│   │   ├── logs/
│   │   │   HEAD
│   │   ├── objects/
│   │   ├── refs/
│   ├── cache/
│   │   048604269e6bb3c73369be4aa05b47d4.pkl
│   │   09770b36e2a0a15c231138c003d001c4.pkl
│   │   09a81e99e9dcf0c30f4a148e94b90f05.pkl
│   │   0f37e087a2cca723f88192e1acd20507.pkl
│   │   0f3a114c024193a616481f342a389e3f.pkl
│   │   14143117d11207daf966d77ca6bae0ec.pkl
│   │   1b6cf806e0a2544dd1ca310489845c7b.pkl
│   │   1d6a2e02c36bced5f8df119fa4b52c4b.pkl
│   │   1ea8151324c1e98c98aa792fbc204fe2.pkl
│   │   200efcdccdd83084bbeaa38bd1bd3024.pkl
│   │   22259bb3fa071e422b337cc7442cb3b0.pkl
│   │   26cd0be6fbfac9c901dbf545ad29b9a1.pkl
│   │   37542e30d68cafe70ac51138e3de71ca.pkl
│   │   41dd9ccf83f4f922e1366c6018272675.pkl
│   │   4f717e130381a9db86728cd112efe911.pkl
│   │   5681243c7ce97d6a08066873b4c52c80.pkl
│   │   5ef1ad5bfe69fc4cabfcf37c9b1d65af.pkl
│   │   660f9d0d4e414a0766e0e86b80d9e1d0.pkl
│   │   6c92331e1676fcf85a4af1d17c911d3e.pkl
│   │   6d895672886b2abb62d254f1c39f0911.pkl
│   │   79cfb1b0694565d076ddbbc95be7a1c9.pkl
│   │   8e8ee95bcac7b14cf0cf45ef17c1d4f7.pkl
│   │   9308ef6f46f719516d304d5a48d6f1bf.pkl
│   │   a31ea451810728222168f9377057fa96.pkl
│   │   aa8effa202f8bb86f5aeed0c4ed209db.pkl
│   │   c8f108b7470ea7fea972aedff49816f0.pkl
│   │   e67c3904498d73681be226e22db6f638.pkl
│   │   e9fb99e1d85114ee95380710249b0f96.pkl
│   │   ebde91ddfacc0316323b7b09476bd49f.pkl
│   │   ff3fe477f56205a4fdd54ba285fc2e17.pkl
│   ├── database/
│   │   connection.py
│   │   crud.py
│   │   models.py
│   │   __init__.py
│   │   ├── __pycache__/
│   │   │   connection.cpython-311.pyc
│   │   │   crud.cpython-311.pyc
│   │   │   models.cpython-311.pyc
│   │   │   __init__.cpython-311.pyc
│   ├── models/
│   │   aapl_lstm_model.pkl
│   │   amzn_lstm_model.pkl
│   │   cost_lstm_model.pkl
│   │   googl_lstm_model.pkl
│   │   msft_lstm_model.pkl
│   │   tsla_lstm_model.pkl
│   │   voo_lstm_model.pkl
│   │   ^ftse_lstm_model.pkl
│   │   ^gdaxi_lstm_model.pkl
│   │   ^gspc_lstm_model.pkl
│   │   ^hsi_lstm_model.pkl
│   │   ^n225_lstm_model.pkl
│   │   ^vix_lstm_model.pkl
│   ├── routes/
│   │   analytics.py
│   │   auth.py
│   │   market_api.py
│   │   pages.py
│   │   prediction_routes.py
│   │   profile.py
│   │   search.py
│   │   stock_api.py
│   │   websocket.py
│   │   __init__.py
│   │   ├── __pycache__/
│   │   │   ai_prediction.cpython-311.pyc
│   │   │   analytics.cpython-311.pyc
│   │   │   auth.cpython-311.pyc
│   │   │   learning.cpython-311.pyc
│   │   │   market_api.cpython-311.pyc
│   │   │   pages.cpython-311.pyc
│   │   │   prediction_routes.cpython-311.pyc
│   │   │   profile.cpython-311.pyc
│   │   │   search.cpython-311.pyc
│   │   │   stock_api.cpython-311.pyc
│   │   │   stock_routes.cpython-311.pyc
│   │   │   websocket.cpython-311.pyc
│   │   │   __init__.cpython-311.pyc
│   ├── scalers/
│   ├── services/
│   │   market_service.py
│   │   prediction_service.py
│   │   stock_chart.py
│   │   stock_service.py
│   │   websocket_manager.py
│   │   ├── __pycache__/
│   │   │   learning_service.cpython-311.pyc
│   │   │   market_service.cpython-311.pyc
│   │   │   prediction_service.cpython-311.pyc
│   │   │   stock_service.cpython-311.pyc
│   │   │   websocket_manager.cpython-311.pyc
│   ├── static/
│   │   ├── css/
│   │   ├── img/
│   │   ├── js/
│   ├── templates/
│   │   about.html
│   │   analytics.html
│   │   base.html
│   │   dashboard.html
│   │   index.html
│   │   learning.html
│   │   login.html
│   │   module.html
│   │   Profile.html
│   │   sample.html
│   │   signup.html
│   ├── utils/
│   │   auth_utils.py
│   │   __init__.py
│   │   ├── __pycache__/
│   │   │   auth_utils.cpython-311.pyc
│   │   │   __init__.cpython-311.pyc
│   ├── venv/
│   │   pyvenv.cfg
│   │   ├── Include/
│   │   ├── Lib/
│   │   ├── Scripts/
│   │   │   activate
│   │   │   activate.bat
│   │   │   Activate.ps1
│   │   │   autopep8.exe
│   │   │   black.exe
│   │   │   blackd.exe
│   │   │   deactivate.bat
│   │   │   dotenv.exe
│   │   │   f2py.exe
│   │   │   fastapi.exe
│   │   │   flask.exe
│   │   │   fonttools.exe
│   │   │   httpx.exe
│   │   │   import_pb_to_tensorboard.exe
│   │   │   markdown-it.exe
│   │   │   markdown_py.exe
│   │   │   normalizer.exe
│   │   │   numpy-config.exe
│   │   │   pip.exe
│   │   │   pip3.11.exe
│   │   │   pip3.exe
│   │   │   pwiz.py
│   │   │   pycodestyle.exe
│   │   │   pyftmerge.exe
│   │   │   pyftsubset.exe
│   │   │   pygmentize.exe
│   │   │   pyrsa-decrypt.exe
│   │   │   pyrsa-encrypt.exe
│   │   │   pyrsa-keygen.exe
│   │   │   pyrsa-priv2pub.exe
│   │   │   pyrsa-sign.exe
│   │   │   pyrsa-verify.exe
│   │   │   python.exe
│   │   │   pythonw.exe
│   │   │   sample.exe
│   │   │   saved_model_cli.exe
│   │   │   tensorboard.exe
│   │   │   tflite_convert.exe
│   │   │   tf_upgrade_v2.exe
│   │   │   toco.exe
│   │   │   ttx.exe
│   │   │   uvicorn.exe
│   │   │   wheel.exe
│   │   ├── share/
│   ├── __pycache__/
│   │   main.cpython-311.pyc


## 🔧 Setup Instructions

```bash
# Clone the repo
git clone https://github.com/rahuulsharmaa/StockMaster.git
cd StockMaster

# Set up virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
uvicorn main:app --reload


🌐 Tech Stack
Python

FastAPI

Jinja2

PostgreSQL

SQLAlchemy / asyncpg

HTML/CSS/JS
**Checksum**
SHA256: 574F2FA2B43012FA25FCA4FCDB36BD7C6BCCDD0AF4242F6C0E4C8633C5A072AE


### For Conda
**create env (takes a few minutes)**
conda env create -f environment.yml

**start Jupyter from that env (important)**
conda activate heartenv
jupyter lab   # or jupyter notebook

### For Docker


### For python only/jupyter notebook
**Commands**
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook

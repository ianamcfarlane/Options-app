FROM python:3.11-slim

WORKDIR /home/coder/options

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir yfinance numpy duckdb

# run as nobody:users to match Unraid
RUN useradd -u 99 -g 100 appuser || true
USER 99:100

EXPOSE 8501
CMD ["bash", "-lc", "streamlit run app/app.py --server.address=0.0.0.0 --server.port=8501"]

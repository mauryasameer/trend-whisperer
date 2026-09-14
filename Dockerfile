FROM python:3.12-slim

RUN useradd --create-home --shell /bin/bash app
WORKDIR /home/app/trend-whisperer

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chown -R app:app /home/app/trend-whisperer

USER app

CMD ["python", "-m", "src.app"]

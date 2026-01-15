FROM python:3.10-slim

WORKDIR /app

RUN useradd -m -u 1000 user
RUN chown -R user:user /app
USER user
ENV PATH="/home/user/.local/bin:$PATH"

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . .

CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "7860"]

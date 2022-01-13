FROM python:3.7
COPY requirements-dev.txt ./requirements-dev.txt
RUN pip install -r requirements-dev.txt
RUN pip install thai2transformers==0.1.2 --no-dependencies
ENTRYPOINT ["streamlit", "run"]
CMD ["main.py"]
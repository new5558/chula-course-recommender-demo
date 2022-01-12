FROM python:3.8
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
RUN pip install thai2transformers==0.1.2 --no-dependencies
ENTRYPOINT ["streamlit", "run"]
CMD ["main.py"]
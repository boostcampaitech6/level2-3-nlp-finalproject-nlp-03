FROM python:3.9

# 
WORKDIR /code

# 
COPY ./requirements.txt /code/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 
COPY ./sql_app /code/sql_app

# 
CMD ["uvicorn", "sql_app.main:app", "--host", "0.0.0.0", "--port", "8001"]
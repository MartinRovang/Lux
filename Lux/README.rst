docker build -t lux:0.1 .
docker run -p 8000:8000 --name myapi lux:0.1
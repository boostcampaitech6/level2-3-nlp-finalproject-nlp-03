run : 
	sh script/run_server.sh

up:
	sudo docker rm -f crud_server || true
	sudo docker run -d -p 8001:8001 --name crud_server crud_server

docker_build:
	sudo docker build -t crud_server .
run : 
	sh script/run_client.sh

build_client:
	npm run build
	sudo rm -rf /var/www/html/build/
	sudo cp -r build/ /var/www/html/build/
	sudo systemctl restart nginx.service
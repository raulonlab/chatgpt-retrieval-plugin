# Heroku
# make heroku-login
# make heroku-push

HEROKU_APP = <your app name> 

heroku-push:
	docker buildx build --platform linux/amd64 -t ${HEROKU_APP} .
	docker tag ${HEROKU_APP} registry.heroku.com/${HEROKU_APP}/web
	docker push registry.heroku.com/${HEROKU_APP}/web
	heroku container:release web -a ${HEROKU_APP}

heroku-login:
	heroku container:login

poetry-export-requirements:
	@poetry export -f requirements.txt --output requirements.txt --without-hashes

import-chatnerds:
	# @mkdir -p ./chatnerds_retrieval_plugin
	# @cp -r ../chatnerds/chatnerds_retrieval_plugin/ ./

	rsync -a --exclude '__pycache__' ../chatnerds/chatnerds_retrieval_plugin ./


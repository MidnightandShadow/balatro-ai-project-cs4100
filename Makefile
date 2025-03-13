# TODO: we should have a entrypoint/ folder outside of src for all of our 
# entrypoints
ENTRYPOINT=entrypoints/main.py
PYTHON=./env/bin/python3

all: env

env: requirements.txt
	python3.11 -m venv env \
		&& source env/bin/activate \
		&& pip3 install --upgrade pip \
		&& pip3 install -r requirements.txt

run:
	${PYTHON} ${ENTRYPOINT}

clean:
	rm -rf env

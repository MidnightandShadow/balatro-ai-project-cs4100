# TODO: we should have a entrypoint/ folder outside of src for all of our 
# entrypoints
ENTRYPOINT=entrypoints/main.py
PYTHON=./env/bin/python3
ACTIVATE=source env/bin/activate

all: env

.PHONY: run format clean

env: requirements.txt
	python3.11 -m venv env \
		&& source env/bin/activate \
		&& pip3 install --upgrade pip \
		&& pip3 install -r requirements.txt

run:
	${PYTHON} ${ENTRYPOINT}

format:
	${ACTIVATE} && black .

clean:
	rm -rf env

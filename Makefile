ENTRYPOINT:=entrypoints/main.py
ACTIVATE:=source env/bin/activate

all: env

.PHONY: run format clean all test

env: requirements.txt
	python3 -m venv env \
		&& source env/bin/activate \
		&& pip3 install --upgrade pip \
		&& pip3 install -r requirements.txt

run:
	${ACTIVATE} && python3 ${ENTRYPOINT}

test:
	${ACTIVATE} && pytest .

format:
	${ACTIVATE} && black .

clean:
	rm -rf env

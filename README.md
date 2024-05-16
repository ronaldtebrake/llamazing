# Setup

Uses 

https://ollama.com
https://www.gradio.app/


## Ollama
1. Install ollama (https://ollama.com/) to ensure you can run a LLM locally

```
docker volume create ollama_data
docker run -d --name ollama -p 11434:11434 -v ollama_data:/root/.ollama ollama/ollama
```

2. Verify it's fine

```
docker exec -it ollama /bin/bash #to use the container interactive terminal
ollama --version #0.1.8
```

3. Select your model from Ollama and run it (insider your container) - https://ollama.ai/library, 

for example for using codellama:

```
ollama run codellama
```

4. You can now test it within your container or using the API


Container:
```
ollama run codellama "Write me a function that outputs the fibonacci sequence"
```

API (see port from step 1):
```
curl http://localhost:11434/api/generate -d '{
  "model": "codellama",
  "prompt": "Write me a function that outputs the fibonacci sequence",
  "stream": false
}'
```

## Application that talks with your LLM

### Virtual environment

1. Create virtual environment, make sure your in the project directory 

```
python3 -m venv .venv
```

2. Activate the virtual environment

```
source .venv/bin/activate
```


3. Verify that it uses the .venv environment

```
which python
```

Should output a filepath that includes the .venv (this to ensure we're not messing with the default python)

This to deactivate (don't do that now ;)):
```
deactivate
```

Prepare pip for the virtual env.

```
python3 -m pip install --upgrade pip
python3 -m pip --version
```

### The application itself

In the requirements.txt we have all the packages we need

1. Install packages
```
run pip install -r requirements.txt
```

2. Run the application

```
python app.py

```


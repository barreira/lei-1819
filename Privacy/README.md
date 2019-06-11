# TENSORFLOW PRIVACY

Para criar o ambiente virtual com todas as dependências necessárias:
	
	conda create -p ./env python=3.7 --file req_conda.txt

Ativar o ambiente virtual:

	conda activate ./env

Instalar dependências pip (incluindo TF Privacy):

	pip install -r requirements.txt

Ambiente de execução:

	./run_mnist.sh - executa o ficheiro mnist.py com um número de épocas = (1, 5, 10) valor de batch_size = (128, 256, 512, 1024)
	./run_mnist_tmux.sh - executa o script run_mnist numa sessão tmux (tmux a -t privacy_mnist para visualizar a sessão)
	./run_mnist_half.sh - executa o ficheiro mnist_half.py que efetua os testes descritos no comando anterior mas apenas para metade do dataset mnist
	./run_mnist_half_tmux.sh - executa o script run_mnist_half numa sessão tmux (tmux a -t privacy_mnist_half para visualizar a sessão)


FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04
WORKDIR /app

RUN apt update && apt install sudo -y && sudo apt update && sudo apt upgrade -y

RUN apt-get update && apt-get install -y locales && rm -rf /var/lib/apt/lists/* \
	&& localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8

RUN sudo apt update && sudo apt install software-properties-common -y \
    && sudo add-apt-repository -y 'ppa:deadsnakes/ppa' \
    && sudo apt install python3.9 -y \
    && sudo apt-get install python3.9-dev -y 

RUN sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
    && sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 2 \
    && sudo update-alternatives --config python3 \
    && sudo apt install python3-pip -y 

RUN sudo apt install unzip -y \
    && sudo apt install zip -y

RUN sudo apt-get clean

RUN pip install numpy==1.26.3
RUN pip install pandas==2.0.3
RUN pip install scikit-learn==1.5.2
RUN pip install matplotlib==3.9.3
RUN pip install nltk==3.9.1
RUN pip install spacy==3.8.2
RUN pip install evaluate==0.2.1
RUN pip install rouge==1.0.1
RUN pip install torch==2.5.1
RUN pip install transformers==4.43.3 
RUN pip install datasets==3.1.0
RUN pip install wandb==0.18.7
RUN pip install rouge-score==0.1.2
RUN pip install bert-score==0.3.13
RUN pip install gdown==5.2.0

WORKDIR /app
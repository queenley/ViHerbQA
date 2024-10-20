apt update && apt install sudo -y && sudo apt update && sudo apt upgrade -y

apt-get update && apt-get install -y locales && rm -rf /var/lib/apt/lists/* \
	&& localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8

sudo apt update && sudo apt install software-properties-common -y \
    && sudo add-apt-repository -y 'ppa:deadsnakes/ppa' \
    && sudo apt install python3.9 -y \
    && sudo apt-get install python3.9-dev -y 

sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
    && sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 2 \
    && sudo update-alternatives --config python3 \
    && sudo apt install python3-pip -y 

sudo apt install unzip -y \
    && sudo apt install zip -y

sudo apt-get clean
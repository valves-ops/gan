#!/bin/bash

git config --global user.email "vinicius.alves.contato@gmail.com"
git config --global user.name "Vinicius Alves"

cd /content/drive/MyDrive/colab/repositories/gan

wget https://raw.githubusercontent.com/valves-ops/meditations/master/scripts/alias.sh
source alias.sh
rm alias.sh

pip3 install tensorflow-gan
pip3 install gin-config
pip3 install anytree
pip3 install wandb
pip3 install ipdb
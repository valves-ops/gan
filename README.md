# GAN

GAN training hyperparameter tuning quantitative investigation

## How to use this repository on a Google Colab instance using VSCode
1. Create a new Colab Notebook

2. Connect to the desired runtime (CPU/GPU/TPU) and mount Google Drive

3. Execute this code snippet on a Colab notebook
```python
# Install colab_ssh on google colab
!pip install colab_ssh --upgrade

# Install TF-GAN
!pip install tensorflow-gan

# Setup Git identity
!git config --global user.email "vinicius.alves.contato@gmail.com"
!git config --global user.name "Vinicius Alves"

# Setup aliases
!wget https://raw.githubusercontent.com/valves-ops/meditations/master/scripts/alias.sh
!source alias.sh

# Start Cloudflare Tunnel
from colab_ssh import launch_ssh_cloudflared, init_git_cloudflared
launch_ssh_cloudflared(password="test123")
```

4. User SSH Remote extension on VSCode to connect to the VSCode Remote SSH

5. Open a VSCode terminal and create the following directory
```
mkdir /content/drive/MyDrive/colab/repositories
```

5. Clone this repository with the following command
```
cd /content/drive/MyDrive/colab/repositories
git clone https://{PERSONAL_ACCESS_TOKEN}@github.com/valves-ops/{REPO}.git
```

6. Open the following folder on VSCode:
```
mkdir /content/drive/MyDrive/colab/
```
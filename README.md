# GAN

GAN training hyperparameter tuning quantitative investigation

## How to use this repository on a Google Colab instance using VSCode
1. Create a new Colab Notebook

2. Connect to the desired runtime (CPU/GPU/TPU) and mount Google Drive

3. Execute this code snippet on a Colab notebook
```
# Install colab_ssh on google colab
!pip install colab_ssh --upgrade

from colab_ssh import launch_ssh_cloudflared, init_git_cloudflared
launch_ssh_cloudflared(password="test123")
```

4. User SSH Remote extension on VSCode to connect to the VSCode Remote SSH 

5. Clone this repository with the following command
```
git clone https://{PERSONAL_ACCESS_TOKEN}@github.com/valves-ops/{REPO}.git
```
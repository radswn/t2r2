# enginora
BSc thesis project 

[![Open in Dev Containers](https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/radswn/enginora)

## Description

*TODO*

## Quick start

*TODO*

## Development setup - devcontainers

We will be developing our code using devcontainers. This allows for great isolation of the working environment, while maintaining local files synchronization and git workflow.

Start by installing [the *better* powershell](https://learn.microsoft.com/en-us/powershell/scripting/install/installing-powershell-on-windows?view=powershell-7.3#install-powershell-using-winget-recommended) and setting it up.

### WSL

This guide and the development process itself will depend heavily on the Linux kernel. For that purpose, we need to have Windows Subsystem for Linux installed.

#### General installation

If on Windows 10, install WSL and make sure you're using WSL 2, not 1. WSL2 comes installed with Windows 11. The default Ubuntu version is sufficient, setup tested on Ubuntu 22.04.

Follow these tutorials:

* https://learn.microsoft.com/en-us/windows/wsl/install
* https://learn.microsoft.com/en-us/windows/wsl/setup/environment

#### Git

Setup Git on WSL and best configure it to use the same credentials as on Windows. It comes installed with most of the WSL distros. In case sth doesn't work here - update both [Windows](https://stackoverflow.com/a/48924212) and [WSL](https://www.cyberithub.com/how-to-update-git-to-a-newest-version-on-linux-ubuntu-20-04-lts/) to LTS versions.

* on Windows, run `git config --global credential.helper wincred`
* in WSL, run `git config --global credential.helper "/mnt/c/Program\ Files/Git/mingw64/libexec/git-core/git-credential-wincred.exe"`

This way you should be able to work over HTTP(S) protocol. In a case that you need ssh protocol enabled, configure key pairs accordingly.

If you encounter `fatal: could not open '.git/COMMIT_EDITMSG': Permission denied` 

* run `sudo chown -R {your_username}  .` while in root directory

#### Test the setup

To test that your WSL and git are working

* (in Powershell) restart WSL to make sure the changes are applied: `wsl --shutdown`
* (in WSL) clone this repo: `git clone https://github.com/radswn/enginora.git`

If something doesn't work at the end of the setup and some time passed, consider reinstalling WSL2 to make sure you are using latest versions.

*Note: It's highly recommended to keep this repo in the WSL file system*

### Docker

The next step is to install Docker, our backbone of containerization, on WSL.

> Note: It's not necessary to configure Docker on Windows at all. The installation presented as the *Windows* one requires a Linux kernel anyway, so it has to work perfectly fine being installed only on our WSL distribution.
> 
> *Who needs Windows anyway?*

Follow [this tutorial](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository) or simply run this script provided by Docker.

```
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

### Visual Studio Code

It's highly recommended to develop code using VSCode, as it has great support for Dev Containers.

First, install Visual Studio Code either

* manually from https://code.visualstudio.com/download or
* `winget install -e --id Microsoft.VisualStudioCode`   

Then, install required extensions listed in `.\vscode\extensions.json`. VSC should automatically prompt to install them as soon as the project is opened.

### Optional - enable CUDA in containers

This part is for those that would like to train their models locally, using CUDA-enabled graphics card, inside the container.

1. Download and install [Windows CUDA drivers](https://www.nvidia.com/Download/index.aspx) or [the whole CUDA Toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64)
2. Make sure that your docker is working in WSL
3. Install NVIDIA Container Toolkit in WSL by following [this guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#setting-up-nvidia-container-toolkit)

> You should be able to check whether it's working at any point in time by running `nvidia-smi` command, be it in Powershell, WSL or Docker container.
> If there are no errors and you can see usage of you graphics card - all good.

More detailed tutorial available [here](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute#setting-up-nvidia-cuda-with-docker).

### Test the whole setup

Copy `.devcontainer/devcontainer.json.template` and rename the copy to `.devcontainer/devcontainer.json`. This way you can easily customize your setup and not mess with the desired original state.

Now, before running, in `.devcontainer/devcontainer.json`:

* add "_dev" to the `REQ_FILE` arg if you wish to install additional libraries used only by the repository contributors (you may as well run `pip -r requirements_dev.txt`)
* add lines with `"--gpus"` and `"all"` (in this order) in the `runArgs` attribute if you're going to use CUDA gpus in the container
* add to `customizations.vscode.extensions` any extension IDs that you would like to be automatically installed with the container
* change `USER_UID` and `USER_GID` if your WSL IDs differ from 1000 (you can check them with `id -u` and `id -g`, respectively)


To verify that the whole procedure succeeded, open the repository in WSL by either
* go to project's directory and type `code .`
* open Windows VS Code installation and click the blue icon in bottom-left corner, then choose `Connect to WSL` and navigate to your project folder

When opening, VS Code should automatically prompt with an option to *Reopen folder to develop in a container*. In case it doesn't happen, click `F1` / use `Ctrl+Shift+P` and search for *Dev Containers: Rebuild and Reopen in Container*. 

If running for the first time, it will take several minutes to build the image and start the container. Click `show logs` to monitor the process.

If no errors appear (bold assumption), a new window should open with *Dev Container: Enginora project* in the bottom-left corner.

FROM python:3.9-slim

RUN apt-get update \
  && apt-get install -y git git-lfs sudo vim

ARG REQ_FILE=requirements_dev.txt
ARG USERNAME=user
ARG USER_UID
ARG USER_GID

RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -ms /bin/bash ${USERNAME} \
    && echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

USER ${USERNAME}

ENV PYTHONPATH '${PYTHONPATH}:/workspaces/enginora/src:/home/${USERNAME}/.local/bin'

COPY ${REQ_FILE} /app/${REQ_FILE}

RUN pip install --upgrade pip && \
  pip install -r /app/${REQ_FILE} && \
  sudo rm -r /app

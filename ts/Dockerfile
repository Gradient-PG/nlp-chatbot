FROM pytorch/torchserve:latest-gpu

WORKDIR /ts
# copy torchserve resources
ADD . /ts
# run script configuring torchserve
CMD /ts/run.sh
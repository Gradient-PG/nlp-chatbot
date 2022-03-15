torchserve --model-store model-store --workflow-store wf-store --ncs --ts-config config.properties 
# && curl -X POST http://127.0.0.1:8081/workflows?url=wf.war
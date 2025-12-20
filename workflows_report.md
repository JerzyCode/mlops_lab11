## Exercise 1

Visible in the pyproject.toml file.


## Exercise 2

After road of debugging, the successful pipeline workflow is under the url: https://github.com/JerzyCode/mlops_lab11/actions/runs/20334824691/job/58418696901


## Exercise 3


- added helper scripts
- refactored app
- refactored Dockerfiles

- builded docker dev image

```bash
jerzy-boksa@jerzyb-laptop:~/Programming/Projects/university/term_3/mlops_lab11$ docker image ls
IMAGE                                         ID             DISK USAGE   CONTENT SIZE   EXTRA
sentiment-onnx:latest                         fa8faeb2cde9        623MB        189MB        
```

Noticed the massive reduction of the size. Previous docker image had 1.33GB size.

Application has starded successfully.



## Exercise 4


- added secrets to github
- created ECR repository
- added steps to the workflow.yaml
- added mangum for aws lambda
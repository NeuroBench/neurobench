# Staging tasks to the NeuroBench benchmarking suite
This is the area where anyone in the neuromorphic community can implement their own custom tasks. 
The most used tasks before every release of a new version of NeuroBench will be considered for addition to the official framework. 
Tasks that are selected to be included in a release will be thoroughly tested and validated and moved to the official list of tasks found int he tasks folder.

We strongly recommend you to adhere to the conventions used in the official tasks and to clearly document the usage of your task.
The file structure used in the official tasks is:
```
├── name_task
│   ├── model_data
│   │   ├── parameter_file_of_model.pth
│   ├── model_specification.py
│   ├── training.py
│   ├── benchmark.py
|   ├── task_tutorial.ipynb
```
If possible, it is preferred that the data can be automatically downloaded with a script, else there should be a clear tutorial on how users can acquire the required dataset. To keep the benchmark framework lightweight, datasets will not be included in releases.


<ins>**Disclaimer:**</ins> tasks in this folder are **not** moderated.

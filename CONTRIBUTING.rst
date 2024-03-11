Contribution Guidelines
=======================

As a community-driven project, we welcome further development from the community. If you are interested in developing extensions to features, programming frameworks, or metrics and tasks, please follow these steps:

1. Discuss Your Idea
--------------------

* First, check our [issue tracker](https://github.com/NeuroBench/neurobench/issues) to see if there is an existing issue related to your idea.
* Look for tasks associated with a specific milestone like `v1.0`, tagged as `help wanted`, or without any assignees.
* If you can't find a related issue, open a new one to discuss your idea.

2. Get Started
--------------

* Fork the project repository to your own GitHub account, clone your fork and configure the remotes.
    .. code-block:: bash

        # Clone the repository from your personal fork into the current directory
        git clone https://github.com/<your-username>/neurobench.git
        # Go to the recently cloned folder.
        cd neurobench
        # Set the original repository as a remote named "upstream."
        git remote add upstream https://github.com/NeuroBench/neurobench.git

* If it has been some time since you initially cloned, ensure you obtain the most recent updates from the upstream source.:
    .. code-block:: bash

        git checkout <remote-branch-name>
        git pull upstream <remote-branch-name>

* Create a new branch where you'll develop your feature, change or fix. Name it descriptively to reflect the nature of your work.
    .. code-block:: bash

        git checkout -b <your-branch-name>

3. Development
--------------

* Start working on your code. Make sure your work aligns with the project's goals and scope.

4. Testing and Documentation
----------------------------

* We use pytest (https://docs.pytest.org/en/stable/) for testing.
* For documentation, follow the Google docstrings (https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) format.
    .. code-block:: bash

        # To build the documentation
        cd docs
        poetry run make html
        
* Write tests for your feature and place them in the `neurobench/tests` directory.
* Document your code thoroughly.

5. Setting Up Pre-commit
---------------------

1. Install pre-commit on your local machine. If you haven't installed it yet, you can do so using pip:

   .. code-block:: bash

       pip install pre-commit

2. Set up the pre-commit hooks with the following command:

   .. code-block:: bash

       pre-commit install

This will install the git hook scripts in your `.git/` directory.

6. Running Pre-commit
------------------

Before committing your changes, run the pre-commit hooks to ensure your code is formatted and linted according to the project's standards:

.. code-block:: bash

    pre-commit run

If all checks pass, you can proceed to commit your changes. If some checks fail, pre-commit will modify the files when possible (e.g., auto-formatting code) or display errors that you need to fix manually.

7. Open a Pull Request
----------------------

* Commit your code changes with a description of the specific modifications you've made
* Locally merge (or rebase) the upstream remote branch into your branch:

    .. code-block:: bash

        git pull [--rebase] upstream <remote-branch-name>
    
* Push your branch up to your fork:

    .. code-block:: bash

        git push origin <your-branch-name>

*   Open a pull request (PR) to merge your branch into the `dev` branch of the main repository, providing a clear and informative title and description for your PR.

Please don't hesitate to reach out to the project maintainers if you have any questions or need assistance with the contribution process. We appreciate your efforts to enhance our project!
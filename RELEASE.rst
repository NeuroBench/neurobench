Version Release Guidelines
=======================

This document describes the guidelines for releasing new versions of the library. We follow semantic versioning, which means our version numbers have three parts: MAJOR.MINOR.PATCH.

- MAJOR version when you make incompatible API changes
- MINOR version when you add functionality in a backwards-compatible manner
- PATCH version when you make backwards-compatible bug fixes


1. Install the `bump-my-version` package:

    ```
    pip install --upgrade bump-my-version
    ```
--------------------

2.  Create a new branch for the release from dev branch:

    ```
    git checkout -b release/x.y.z
    ```
--------------------

3. Update the version number using the `bump-my-version` command:

    ```
    bump-my-version bump path
    ```
    or
    ```
    bump-my-version bump minor
    ```
    or
    ```
    bump-my-version bump major
    ```
--------------------

4. Commit the changes with the following message and push the changes to the release branch:

    ```
    git commit -m "Bump version: {current_version} → {new_version}"
    ```

    ```
    git push origin release/x.y.z
    ```

--------------------

5. Create a pull request from the release branch to the dev branch.

6. Once the pull request is approved and merged, create a new pull request from the dev branch to the master branch.

7. Once the pull request is approved and merged, create the tag on the main branch to invoke the package publishing workflow:

    ```
    git tag -a x.y.z -m "Release x.y.z"
    ```

    ```
    git push origin
    ```
--------------------

8. Once the tag is pushed, the package publishing workflow will be triggered and the package will be published to the PyPI.

9. Once the package is published, create a new release on GitHub with the tag name and the release notes.


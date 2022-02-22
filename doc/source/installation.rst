..
    Copyright (c) 2022 Blue Brain Project/EPFL

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

Installation
============

``vascpy`` requires Python version 3.7 or higher. It is recommended to create a virtual environment in order to install the package.

Virtualenv setup
^^^^^^^^^^^^^^^^

In following code samples, the prompts ``(venv)$`` and ``$`` are used to indicate that the user virtualenv is *activated* or *deactivated* respectively.

.. code-block:: bash

    $ python3 -mvenv venv       # creates a virtualenv called "venv" in the current directory
    $ source venv/bin/activate  # activates the "venv" virtualenv
    (venv)$                     # now we are in the venv virtualenv

Upgade the ``pip`` version as shown below:

.. code-block:: bash

    (venv)$ pip install --upgrade pip   # Install newest pip inside virtualenv if version too old.

To de-activate the virtualenv run the ``deactivate`` command:

.. code-block:: bash

    (venv)$ deactivate

Note that you do not have to work in the ``venv`` directory. This is where python packages will
get installed, but you can work anywhere on your file system, as long as you have activated the
``virtualenv``.

Installation options
^^^^^^^^^^^^^^^^^^^^

Install from the official PyPI server
-------------------------------------

Install the latest release:

.. code-block:: bash

    pip install vascpy

Install a specific version:

.. code-block:: bash

    pip install vascpy==0.1.0

Install from git
----------------

Install a particular release:

.. code-block:: bash

    pip install git+https://github.com/BlueBrain/vascpy.git@v0.1.0

Install the latest version:

.. code-block:: bash

    pip install git+https://github.com/BlueBrain/vascpy.git


Install from source
-------------------

Clone the repository and install it:

.. code-block:: bash

    git clone https://github.com/BlueBrain/vascpy.git
    pip install -e ./vascpy

This installs ``vascpy`` into your ``virtualenv`` in "editable" mode. That means
that changes made to the source code after the installation procedure are seen by the
installed package. To install in read-only mode, omit the ``-e``.


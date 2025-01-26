# Dorie cookbook

Example code for building applications with Dorie, with an emphasis on more applied and end-to-end examples than contained in the repo documentation.

Notebook | Description
:- | :-
[Synthetic_Data_Generation.ipynb](https://github.com/loaizasteven/dorie/tree/master/cookbook/Synthetic_Data_Generation.ipynb) | Generate synthetic data for an insurance company using OpenAI.
[Intent_Detection_LoRA.ipynb)](https://github.com/loaizasteven/dorie/tree/master/cookbook/Intent_Detection_LoRA.ipynb) | Train Intent Detection model using Low-Rank Adapatation with Sequence classification tasks.

## Jupyter Kernel Installation
### Using `requirements.txt` to Create the Virtual Environment

To create a virtual environment using the `requirements.txt` file, follow these steps:

1. Navigate to the root directory of the repository:
    ```sh
    cd /path/to/your/project
    ```
2. Create a virtual environment:
    ```sh
    python3 -m venv myenv
    ```
3. Activate the virtual environment:
    - On macOS and Linux:
        ```sh
        source myenv/bin/activate
        ```
    - On Windows:
        ```sh
        .\myenv\Scripts\activate
        ```
4. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

### Using the Shell Script to Set Up the Environment

Alternatively, you can use the provided shell script to set up the environment. See [virtualenv.sh](../libs/dorie/virtualenv.sh).

### Creating the Jupyter Kernel

After setting up the virtual environment, create a Jupyter kernel to use with the notebooks:

1. Install the `ipykernel` package if not already installed:
    ```sh
    pip install ipykernel
    ```
2. Create a new Jupyter kernel:
    ```sh
    python -m ipykernel install --user --name=myenv --display-name "Python (myenv)"
    ```

### Changing the Kernel in Jupyter Notebook

To change the kernel of a Jupyter Notebook (.ipynb file) to use the virtual environment:

1. Open your Jupyter Notebook in VS Code.
2. Click on the kernel picker in the top right corner of the notebook.
3. Choose "Select Another Kernel".
4. Select the kernel with the display name "Python (myenv)".

You can also use the `Notebook: Select Notebook Kernel` command to select the kernel.

This will allow you to use the virtual environment as the kernel for your Jupyter Notebook.
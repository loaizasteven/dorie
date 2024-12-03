# Run Python script to install huggingface_hub[cli]
PYTHON_EXECUTABLE=$(which python3 || which python)
COMMAND=$1
ADDITIONAL_ARGS="$@"


case $COMMAND in
    revision)
        echo -e "\n\033[1;31mAdd specific revisions as command line args for deletion: \033[0m \033[1;37mbash hf_management.sh delete\033[0m\n"
        $PYTHON_EXECUTABLE - <<EOF
from huggingface_hub.utils import scan_cache_dir

hf_cache_info = scan_cache_dir()
print(hf_cache_info.export_as_table(verbosity=1))
EOF
        ;;
    delete)
        $PYTHON_EXECUTABLE - <<EOF
from huggingface_hub.utils import scan_cache_dir
hf_cache_info = scan_cache_dir()
deletion_list = "$ADDITIONAL_ARGS".split()[1:]

for model_id in deletion_list:
    print(f"Deleting model {model_id}")
    scan_cache_dir().delete_revisions(model_id).execute()
EOF
    ;;
    *)
    # Helper comments for huggingface cli management
    echo -e "\n\033[1;34mInstall huggingface_hub[cli] to run the following command on the terminal to understand the memory usage of huggingface cache and how to clear space.\033[0m\n"

    echo -e "\033[1;32mTo check memory usage of the cache run:\033[0m \033[1;37mhuggingface-cli cache\033[0m"
    echo -e "\033[1;32mTo clean the cache run:\033[0m \033[1;37mhuggingface-cli delete-cache\033[0m\n"

    echo -e "\033[1;34mSee more at:\033[0m \033[1;36mhttps://huggingface.co/docs/huggingface_hub/en/guides/manage-cache#clean-your-cache\033[0m \n"

    echo -e "\n\033[1;34mCurrent disk space used by loaded models in HF, to interactively remove them run with arg command-line argument \033[0m \033[1;37mrevision\033[0m\n"
    $PYTHON_EXECUTABLE - <<EOF
from huggingface_hub.utils import scan_cache_dir

hf_cache_info = scan_cache_dir()
print(hf_cache_info.export_as_table(verbosity=0))
EOF
    ;;
esac
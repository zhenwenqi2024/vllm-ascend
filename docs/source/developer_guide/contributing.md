# Contributing

## Building and testing
It's recommended to set up a local development environment to build and test
before you submit a PR.

### Prepare environment and build

Theoretically, the vllm-ascend build is only supported on Linux because
`vllm-ascend` dependency `torch_npu` only supports Linux.

But you can still set up dev env on Linux/Windows/macOS for linting and basic
test as following commands:

```bash
# Choose a base dir (~/vllm-project/) and set up venv
cd ~/vllm-project/
python3 -m venv .venv
source ./.venv/bin/activate

# Clone vllm code and install
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -r requirements/build.txt
VLLM_TARGET_DEVICE="empty" pip install .
cd ..

# Clone vllm-ascend and install
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
# install system requirement
apt install -y gcc g++ cmake libnuma-dev
# install project requirement
pip install -r requirements-dev.txt

# Then you can run lint and mypy test
bash format.sh

# Build:
# - only supported on Linux (torch_npu available)
# pip install -e .
# - build without deps for debugging in other OS
# pip install -e . --no-deps
# - build without custom ops
# COMPILE_CUSTOM_KERNELS=0 pip install -e .

# Commit changed files using `-s`
git commit -sm "your commit info"
```

### Testing

Although vllm-ascend CI provide integration test on [Ascend](https://github.com/vllm-project/vllm-ascend/blob/main/.github/workflows/vllm_ascend_test.yaml), you can run it
locally. The simplest way to run these integration tests locally is through a container:

```bash
# Under Ascend NPU environment
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend

export IMAGE=vllm-ascend-dev-image
export CONTAINER_NAME=vllm-ascend-dev
export DEVICE=/dev/davinci1

# The first build will take about 10 mins (10MB/s) to download the base image and packages
docker build -t $IMAGE -f ./Dockerfile .
# You can also specify the mirror repo via setting VLLM_REPO to speedup
# docker build -t $IMAGE -f ./Dockerfile . --build-arg VLLM_REPO=https://gitee.com/mirrors/vllm

docker run --rm --name $CONTAINER_NAME --network host --device $DEVICE \
           --device /dev/davinci_manager --device /dev/devmm_svm \
           --device /dev/hisi_hdc -v /usr/local/dcmi:/usr/local/dcmi \
           -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
           -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
           -ti $IMAGE bash

cd vllm-ascend
pip install -r requirements-dev.txt

pytest tests/
```


### Run doctest

vllm-ascend provides a `vllm-ascend/tests/e2e/run_doctests.sh` command to run all doctests in the doc files.
The doctest is a good way to make sure the docs are up to date and the examples are executable, you can run it locally as follows:

```{code-block} bash
   :substitutions:

# Update DEVICE according to your device (/dev/davinci[0-7])
export DEVICE=/dev/davinci0
# Update the vllm-ascend image
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
--name vllm-ascend \
--device $DEVICE \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /root/.cache:/root/.cache \
-p 8000:8000 \
-it $IMAGE bash

# Run doctest
/vllm-workspace/vllm-ascend/tests/e2e/run_doctests.sh
```

This will reproduce the same environment as the CI: [vllm_ascend_doctest.yaml](https://github.com/vllm-project/vllm-ascend/blob/main/.github/workflows/vllm_ascend_doctest.yaml).


## DCO and Signed-off-by

When contributing changes to this project, you must agree to the DCO. Commits must include a `Signed-off-by:` header which certifies agreement with the terms of the DCO.

Using `-s` with `git commit` will automatically add this header.

## PR Title and Classification

Only specific types of PRs will be reviewed. The PR title is prefixed appropriately to indicate the type of change. Please use one of the following:

- `[Attention]` for new features or optimization in attention.
- `[Communicator]` for new features or optimization in communicators.
- `[ModelRunner]` for new features or optimization in model runner.
- `[Platform]` for new features or optimization in platform.
- `[Worker]` for new features or optimization in worker.
- `[Core]` for new features or optimization  in the core vllm-ascend logic (such as platform, attention, communicators, model runner)
- `[Kernel]` changes affecting compute kernels and ops.
- `[Bugfix]` for bug fixes.
- `[Doc]` for documentation fixes and improvements.
- `[Test]` for tests (such as unit tests).
- `[CI]` for build or continuous integration improvements.
- `[Misc]` for PRs that do not fit the above categories. Please use this sparingly.

:::{note}
If the PR spans more than one category, please include all relevant prefixes.
:::

## Others

You may find more information about contributing to vLLM Ascend backend plugin on [<u>docs.vllm.ai</u>](https://docs.vllm.ai/en/latest/contributing/overview.html).
If you find any problem when contributing, you can feel free to submit a PR to improve the doc to help other developers.

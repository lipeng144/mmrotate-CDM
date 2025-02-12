
## Introduction

We have integrated the CDM encoding method into the mmrotate platform.The corresponding papers for this code are 
"Peng Li, Cunqian Feng, Weike Feng,Xiaowei Hu. Oriented Bounding Box Representation Based on Continuous Encoding in Oriented SAR Ship Detection[J]. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing. "

The master branch works with **PyTorch 1.6+**.

<details open>
<summary><b>Major Features</b></summary>

- **Support multiple angle representations**

  MMRotate provides three mainstream angle representations to meet different paper settings.

- **Modular Design**

  We decompose the rotated object detection framework into different components,
  which makes it much easy and flexible to build a new model by combining different modules.

- **Strong baseline and State of the art**

  The toolbox provides strong baselines and state-of-the-art methods in rotated object detection.

</details>


## Installation

Our method is in the 'configs/cdm' folder. Please refer to [Installation](https://mmrotate.readthedocs.io/en/1.x/get_started.html) for more detailed instruction.

## Getting Started

Please see [Overview](https://mmrotate.readthedocs.io/en/1.x/overview.html) for the general introduction of MMRotate.

For detailed user guides and advanced guides, please refer to [documentation](https://mmrotate.readthedocs.io/en/1.x/):

- User Guides
  - [Train & Test](https://mmrotate.readthedocs.io/en/1.x/user_guides/index.html#train-test)
    - [Learn about Configs](https://mmrotate.readthedocs.io/en/1.x/user_guides/config.html)
    - [Inference with existing models](https://mmrotate.readthedocs.io/en/1.x/user_guides/inference.html)
    - [Dataset Prepare](https://mmrotate.readthedocs.io/en/1.x/user_guides/dataset_prepare.html)
    - [Test existing models on standard datasets](https://mmrotate.readthedocs.io/en/1.x/user_guides/train_test.html)
    - [Train predefined models on standard datasets](https://mmrotate.readthedocs.io/en/1.x/user_guides/train_test.html)
    - [Test Results Submission](https://mmrotate.readthedocs.io/en/1.x/user_guides/test_results_submission.html)
  - [Useful Tools](https://mmrotate.readthedocs.io/en/1.x/user_guides/index.html#useful-tools)
- Advanced Guides
  - [Basic Concepts](https://mmrotate.readthedocs.io/en/1.x/advanced_guides/index.html#basic-concepts)
  - [Component Customization](https://mmrotate.readthedocs.io/en/1.x/advanced_guides/index.html#component-customization)
  - [How to](https://mmrotate.readthedocs.io/en/1.x/advanced_guides/index.html#how-to)
  
## Data Preparation

Prepare the dataset according to the format in mmrotate and put it into the 'tools/data folder'.


## License

This project is released under the [Apache 2.0 license](LICENSE).

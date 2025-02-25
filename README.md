## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Acknowledgements](#acknowledgements)

## Overview
Repana was developed based on the [repeng](https://github.com/vgel/repeng) library for research purposes. It is intended for doing representation analysis of large language models. `Repana` allows you to insert a (or more) `ControlBlock` into a standard torch transformer model creating a `ControlModel` instance. You can then "read" model representations, use these to derive a `ControlVector`, which can be applied to the `ControlModel` during inference.

## Features
- ControlModel
- ControlVector
  - Reading vector
  - Contrast vector
  - PCA-based contrast vector

## Installation
```bash
pip install repana
```

## Acknowledgements
This library is based off of the [repeng](https://github.com/vgel/repeng) library (MIT License) developed by Theia Vogel. If you want to merely train control vectors fast, we suggest taking a look at that great library!


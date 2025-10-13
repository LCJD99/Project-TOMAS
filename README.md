# Project-TOMAS: Tool-augmented Optimal Multi-task DNN Scheduling

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-in%20progress-orange.svg)]()

**TOMAS** (which stands for **T**ool-augmented **O**ptimal **M**ulti-task DNN **S**cheduling) is a research framework for exploring the use of Tool-Augmented Large Language Models (LLMs) to optimize the scheduling of multiple Deep Neural Network (DNN) tasks in resource-constrained environments.

[English Version](./README.md) | [中文版](./README_zh.md)

---

## Table of Contents

- [Introduction](#introduction)
- [The Core Problem](#the-core-problem)
- [Our Approach](#our-approach)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Installation Guide](#installation-guide)
- [Quick Start](#quick-start)
- [Reproducing Experiments](#reproducing-experiments)
- [Contribution Guidelines](#contribution-guidelines)
- [How to Cite](#how-to-cite)
- [License](#license)

## Introduction

With the proliferation of edge computing and IoT devices, efficiently executing multiple DNN models (e.g., for concurrent image recognition, voice processing, and sensor data analysis) on resource-constrained platforms has become critically important. Traditional scheduling algorithms (like FIFO, Round Robin, or priority-based methods) often struggle to adapt to dynamic workloads and complex hardware constraints, leading to suboptimal performance and resource wastage.

**Project-TOMAS** explores a novel scheduling paradigm: leveraging a Large Language Model (LLM) as the decision-making "brain." By empowering the LLM with the ability to call "tools"—such as querying device status, profiling model performance, or predicting execution times—we construct an intelligent scheduling agent capable of understanding complex objectives and dynamically generating near-optimal scheduling policies.

This repository contains all the experimental code, simulation environments, and result analyses for our accompanying research paper.

## The Core Problem

## Our Approach

## Key Features

## System Architecture

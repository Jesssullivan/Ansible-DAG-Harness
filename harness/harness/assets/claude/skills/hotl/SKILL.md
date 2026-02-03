---
name: hotl
description: Human-on-the-loop autonomous batch processing for Ansible roles
user-invocable: true
argument-hint: "[start|stop|status]"
allowed-tools: Read, Grep, Glob, Bash, Task
---

# HOTL Mode (Human-on-the-Loop)

Manage autonomous batch processing of Ansible role box-up workflows.

## Usage

### Start HOTL Mode
```bash
harness hotl start
```

### Check Status
```bash
harness hotl status
```

### Stop HOTL Mode
```bash
harness hotl stop
```

## Overview

HOTL mode processes roles in wave order, automatically executing the box-up-role
workflow for each role. It pauses at `human_approval` checkpoints for review.

## Reviewing Paused Workflows

```bash
# See what's waiting for approval
harness hotl status

# Approve a specific role
harness resume <execution-id> --approve

# Reject with reason
harness resume <execution-id> --reject --reason "Needs more tests"
```

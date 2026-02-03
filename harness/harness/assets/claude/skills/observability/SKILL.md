---
name: observability
description: Debug and monitor platform services, IIS, and application health
user-invocable: true
argument-hint: "[debug-platform-health|debug-ems-logs|debug-iis-errors|debug-hrtk-status]"
allowed-tools: Read, Grep, Glob, Bash, Task
---

# Observability Commands

Debug and monitor infrastructure health.

## Available Commands

### Platform Health
```bash
harness check
npm run platform:health
```

### Application Logs
```bash
npm run logs:eventlog:application
npm run logs:errors:all
```

### IIS Errors
```bash
npm run desktop:health:server
npm run logs:eventlog:system
```

### SQL Agent Jobs
```bash
npm run jobs:sql-agent
npm run jobs:sql-agent:failures
```

### Grafana Alloy Status
```bash
npm run alloy:health
npm run alloy:status
npm run alloy:metrics
```

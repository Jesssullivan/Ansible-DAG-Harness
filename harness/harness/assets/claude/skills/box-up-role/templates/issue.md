## Summary

Deploy and test the `{{ role_name }}` Ansible role for EMS infrastructure.

**Wave**: {{ wave_number }} ({{ wave_name }})

## Description

{{ role_description }}

## Acceptance Criteria

- [ ] Molecule tests pass against {{ deploy_target | default('test target') }}
- [ ] Role deploys successfully via `npm run deploy:{{ deploy_target }} -- --tags {{ role_tags }}`
- [ ] No regressions in dependent roles
- [ ] All idempotency checks pass (second run shows no changes)
- [ ] Credentials verified in KeePassXC

## Prerequisites

### Credentials Required
{% for cred in credentials %}
- `{{ cred.entry }}` - {{ cred.purpose }}{% if cred.base58 %} (base58 encoded){% endif %}
{% endfor %}

### Role Dependencies
{% if explicit_deps %}
These roles must be deployed first:
{% for dep in explicit_deps %}
- `{{ dep }}`
{% endfor %}
{% else %}
No explicit dependencies (foundation role).
{% endif %}

{% if implicit_deps %}
### Implicit Dependencies (variable references)
{% for dep in implicit_deps %}
- `{{ dep }}`
{% endfor %}
{% endif %}

## Test Instructions

### 1. Molecule Test (isolated)
```bash
npm run molecule:role --role={{ role_name }}
```

### 2. Deploy to Test Target
```bash
npm run deploy:{{ deploy_target }} -- --tags {{ role_tags }}
```

### 3. Verify Deployment
```bash
# Check service status (if applicable)
npm run platform:health

# Check IIS status (if applicable)
npm run desktop:health:server
```

## Files Changed

- `ansible/roles/{{ role_name }}/` - Role implementation
- `package.json` - npm scripts for deployment
- `ansible/site.yml` - Tag integration

## Metadata

| Field | Value |
|-------|-------|
| Wave | {{ wave_number }} |
| Iteration | {{ iteration_name }} |
| Assignee | {{ assignee }} |
| Labels | {{ labels | join(', ') }} |

---

/label ~role ~ansible ~molecule
/assign @{{ assignee }}

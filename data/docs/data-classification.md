# Data Classification Policy

Acme Financial (fictional company) - sample document for this demo.

## Classification levels

Public: information approved for release, such as published annual reports and marketing material.

Internal: day-to-day business information, such as org charts and internal procedures. Must not be shared outside the company.

Confidential: customer data, account balances, transaction history and employee records. Access is limited to people who need it for their role.

Restricted: card numbers, authentication secrets, encryption keys and regulator correspondence. Must be encrypted at rest and in transit, and every access is logged.

## Using AI tools

Confidential and Restricted data must never be pasted into public AI chatbots or external cloud AI services. Only approved on-premises AI systems may process Confidential data, and personal data must be masked before it reaches a model.

## Retention

Customer transaction records are kept for 8 years. Application logs are kept for 1 year. Audit logs are kept for 7 years and must be tamper-evident.

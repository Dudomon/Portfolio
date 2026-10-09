---
doc_id: ERP-001
title: ERP Access and Segregation of Duties
category: erp
owner: ERP Security and Internal Controls
effective_date: 2025-04-01
---

# ERP Access and Segregation of Duties

Access to the ERP system is granted through business roles. Each role bundles the
transactions needed for a job function and is owned by a named role owner in the
business.

## Requesting access

Access requests are submitted in the identity portal and must name the business role,
the company codes and the business justification. Requests are approved by the
employee's manager and the role owner. Direct assignment of individual transactions
outside a business role is not allowed.

## Segregation of duties

The following combinations are conflicting and may not be held by the same user:

- Creating or changing supplier master data and processing supplier payments.
- Creating purchase orders and posting goods receipts for the same purchase orders.
- Posting manual journal entries and approving manual journal entries.
- Maintaining user access and holding any financial posting role.

Exceptions require a documented mitigating control, such as a monthly review of the
user's transactions, approved by Internal Controls. Exceptions are valid for at most
6 months and must be renewed.

## Emergency access

Emergency (firefighter) access may be used for production incidents. It is time-limited
to 24 hours, all activity is logged, and the log is reviewed by the role owner within 5
business days.

## Access reviews

Role owners review all users assigned to their roles every quarter. Access not confirmed
within 15 days of the review start is removed automatically.

## Leavers and movers

Access is removed on the employee's last working day. When an employee changes jobs,
roles that are no longer needed are removed within 5 business days of the move.

---
doc_id: AP-001
title: Invoice Processing and Three-Way Match
category: accounts-payable
owner: Accounts Payable
effective_date: 2025-03-01
---

# Invoice Processing and Three-Way Match

Accounts Payable (AP) processes supplier invoices received at the central AP mailbox or
through the supplier portal. Invoices sent to individual employees must be forwarded to
AP the same day and must not be approved informally by email.

## Three-way match

PO-backed invoices for goods are matched in the ERP system against the purchase order
(price and quantity) and the goods receipt (quantity received). Invoices for services
are matched against the PO and a service entry sheet confirmed by the requester.

### Tolerances

An invoice is posted automatically when the price variance per line is within 2 percent
or USD 100, whichever is lower, and the invoiced quantity does not exceed the received
quantity. Variances outside tolerance block the invoice for payment.

### Blocked invoices

Blocked invoices are routed to the buyer for price variances and to the requester for
quantity variances. They must be resolved within 5 business days. AP never changes the
price on a supplier invoice; a corrected invoice or credit memo is required instead.

## Invoices without a purchase order

Invoices without a PO are only accepted for the categories listed in the No-PO
Exception List (utilities, rent, government fees, and legal settlements). All other
non-PO invoices are returned to the supplier with a request to quote a valid PO number.

## Duplicate invoice controls

The ERP system flags invoices with the same supplier, invoice number and amount, or the
same supplier, amount and invoice date within 60 days. Flagged invoices are reviewed by
an AP analyst before posting.

## Goods receipt timing

Requesters must post goods receipts within 3 business days of physical receipt. Late
goods receipts are the most common cause of blocked invoices and lost early payment
discounts.

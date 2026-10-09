---
doc_id: TAX-002
title: Withholding Tax Procedure
category: tax
owner: Corporate Tax
effective_date: 2025-02-01
---

# Withholding Tax Procedure

This procedure explains when Halvorsen Industries must withhold United States tax from
payments to suppliers and how withholding is handled in the ERP system.

## Payments to foreign suppliers

Payments of US-source income to foreign persons, such as service fees for work performed
in the United States, royalties and certain licensing fees, are subject to withholding
tax (WHT). Payments for goods are generally not subject to withholding.

### Withholding rates

The default WHT rate is 30 percent of the gross payment. A reduced treaty rate applies
only when the supplier has provided a valid Form W-8BEN or W-8BEN-E claiming treaty
benefits, including the supplier's foreign tax identification number.

### Missing or expired forms

If a valid Form W-8 is not on file at the time of payment, the 30 percent default rate
is applied. Forms W-8 expire at the end of the third calendar year after signature. The
ERP system places a payment hold on suppliers whose form expires within 30 days, and
Corporate Tax contacts the supplier for a renewed form.

## Domestic backup withholding

United States suppliers that do not provide a valid Form W-9 with a taxpayer
identification number are subject to backup withholding at 24 percent.

## Form 1099 reporting

Payments of USD 600 or more in a calendar year to United States non-corporate suppliers
for services are reported on Form 1099-NEC. Withholding tax codes in the supplier master
record determine which payments are reportable; buyers must not change them.

## Responsibilities

Corporate Tax owns the WHT configuration and treaty rate table. The Vendor Master Data
Team maintains tax forms on supplier records. AP must not release a payment hold related
to withholding without written approval from Corporate Tax.

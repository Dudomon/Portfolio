# SAP Integration Guide - Brazilian Tax Reform System

## Overview

This guide provides step-by-step instructions for integrating the Brazilian Tax Reform System with SAP ERP (ECC or S/4HANA) to enable IBS/CBS calculations in procurement, sales, and financial processes.

## Architecture

```
SAP ERP                           Tax Reform API
┌─────────────────┐              ┌──────────────────┐
│                 │              │                  │
│  SD Module      │─────────────▶│ Tax Calculation  │
│  (Sales)        │◀─────────────│ Service          │
│                 │              │                  │
│  MM Module      │─────────────▶│ Compliance       │
│  (Procurement)  │◀─────────────│ Validation       │
│                 │              │                  │
│  FI Module      │─────────────▶│ Gap Analysis     │
│  (Finance)      │◀─────────────│ Service          │
│                 │              │                  │
└─────────────────┘              └──────────────────┘
```

## Integration Points

### 1. Pricing Procedure Configuration

#### Create New Condition Types

Execute in SAP transaction **V/06**:

```
Condition Type: ZIBS
Description: IBS - Imposto sobre Bens e Serviços
Condition Class: C (Taxes)
Calculation Type: A (Percentage)
Condition Category: G (Freight/Taxes)
Plus/Minus: A (Plus)
Manual Entries: B (Not possible)
Account Key: MWS
Accrual Key: FR1
```

```
Condition Type: ZCBS
Description: CBS - Contribuição sobre Bens e Serviços
Condition Class: C (Taxes)
Calculation Type: A (Percentage)
Condition Category: G (Freight/Taxes)
Plus/Minus: A (Plus)
Manual Entries: B (Not possible)
Account Key: MW2
Accrual Key: FR2
```

```
Condition Type: ZICT
Description: ICMS Transição (Legacy)
Condition Class: C (Taxes)
Calculation Type: A (Percentage)
Condition Category: G (Freight/Taxes)
Plus/Minus: A (Plus)
```

#### Update Pricing Procedure (V/08)

Add to pricing procedure **ZVATBR** (Brazilian VAT):

```
Step  Cond  Description           From   To    Manual  Requirement
─────────────────────────────────────────────────────────────────
010   PR00  Price                                X
020   SKTO  Cash Discount                        X
100   NETW  Net Value             010    020
...
500   ZIBS  IBS Tax               100           -      50
510   ZCBS  CBS Tax               100           -      50
520   ZICT  ICMS Transition       100           -      51
530   ZPST  PIS Transition        100           -      51
540   ZCFT  COFINS Transition     100           -      51
...
900   MWST  Total Tax             500    540
999   KZWI  Net Price             100    900
```

**Requirements:**
- Req 50: Active only if sy-datum >= '20260101' (Tax Reform start)
- Req 51: Active only if sy-datum >= '20260101' AND transition not complete

### 2. External Tax Calculation (ABAP User-Exit)

#### Implement User-Exit in SD Pricing

**Enhancement:** RV60AFZZ (SD Pricing User-Exits)
**Function Module:** EXIT_SAPLV60A_001

```abap
*----------------------------------------------------------------------*
* User-Exit: Calculate IBS/CBS via External Tax API
*----------------------------------------------------------------------*
FUNCTION z_calculate_tax_reform.

  DATA: lt_tax_request TYPE TABLE OF zs_tax_request,
        ls_tax_request TYPE zs_tax_request,
        lt_tax_result TYPE TABLE OF zs_tax_result,
        ls_tax_result TYPE zs_tax_result,
        lv_http_code TYPE i,
        lv_response TYPE string.

  DATA(lo_http_client) = NEW zcl_http_client( ).
  DATA(lo_json) = NEW zcl_json_parser( ).

  " Build request for each line item
  LOOP AT xkomv ASSIGNING FIELD-SYMBOL(<komv>) WHERE kschl = 'ZIBS' OR kschl = 'ZCBS'.

    READ TABLE xvbap ASSIGNING FIELD-SYMBOL(<vbap>)
      WITH KEY posnr = <komv>-kposn.
    CHECK sy-subrc = 0.

    " Get material master data
    SELECT SINGLE matnr, ncm_code, origin_state
      FROM mara
      WHERE matnr = <vbap>-matnr
      INTO @DATA(ls_material).

    " Prepare API request
    CLEAR ls_tax_request.
    ls_tax_request-base_value = <komv>-kwert.
    ls_tax_request-year = sy-datum(4).
    ls_tax_request-ncm = ls_material-ncm_code.
    ls_tax_request-origin_state = ls_material-origin_state.
    ls_tax_request-destination_state = <vbkd>-state. "Customer state
    ls_tax_request-is_export = COND #( WHEN <vbkd>-country NE 'BR' THEN abap_true ELSE abap_false ).

    APPEND ls_tax_request TO lt_tax_request.

  ENDLOOP.

  " Call external Tax Reform API
  DATA(lv_request_json) = lo_json->serialize( lt_tax_request ).

  lo_http_client->request->set_method( 'POST' ).
  lo_http_client->request->set_uri( '/api/tax/calculate-batch' ).
  lo_http_client->request->set_header_field( name = 'Content-Type' value = 'application/json' ).
  lo_http_client->request->set_header_field( name = 'Authorization' value = 'Bearer {API_TOKEN}' ).
  lo_http_client->request->set_cdata( lv_request_json ).

  " Send HTTP request
  lo_http_client->send(
    EXCEPTIONS
      http_communication_failure = 1
      http_invalid_state = 2 ).

  IF sy-subrc <> 0.
    " Log error and use fallback calculation
    MESSAGE e001(ztax) WITH 'Tax API call failed - using fallback'.
    RETURN.
  ENDIF.

  " Receive response
  lo_http_client->receive(
    EXCEPTIONS
      http_communication_failure = 1
      http_invalid_state = 2 ).

  lv_http_code = lo_http_client->response->get_status( )-code.
  lv_response = lo_http_client->response->get_cdata( ).

  IF lv_http_code = 200.
    " Parse JSON response
    lt_tax_result = lo_json->deserialize( lv_response ).

    " Update pricing conditions
    LOOP AT lt_tax_result INTO ls_tax_result.

      " IBS condition
      READ TABLE xkomv ASSIGNING <komv>
        WITH KEY kschl = 'ZIBS'
                 kposn = ls_tax_result-line_item.
      IF sy-subrc = 0.
        <komv>-kbetr = ls_tax_result-ibs_rate * 10. "SAP format: rate * 10
        <komv>-kwert = ls_tax_result-ibs_amount.
        <komv>-kstat = 'A'. "Active
      ENDIF.

      " CBS condition
      READ TABLE xkomv ASSIGNING <komv>
        WITH KEY kschl = 'ZCBS'
                 kposn = ls_tax_result-line_item.
      IF sy-subrc = 0.
        <komv>-kbetr = ls_tax_result-cbs_rate * 10.
        <komv>-kwert = ls_tax_result-cbs_amount.
        <komv>-kstat = 'A'.
      ENDIF.

      " Legacy ICMS (transition)
      READ TABLE xkomv ASSIGNING <komv>
        WITH KEY kschl = 'ZICT'
                 kposn = ls_tax_result-line_item.
      IF sy-subrc = 0.
        <komv>-kbetr = ls_tax_result-legacy_icms_rate * 10.
        <komv>-kwert = ls_tax_result-legacy_icms_amount.
        <komv>-kstat = COND #( WHEN ls_tax_result-legacy_icms_amount > 0 THEN 'A' ELSE 'I' ).
      ENDIF.

    ENDLOOP.

  ELSE.
    " HTTP error - log and use fallback
    MESSAGE e002(ztax) WITH 'Tax API error code:' lv_http_code.
  ENDIF.

ENDFUNCTION.
```

### 3. Account Determination Configuration

#### Create GL Accounts (FS00)

```
1010301001 - IBS Recoverable (Input Tax)
1010301002 - CBS Recoverable (Input Tax)
2010101001 - IBS Payable (Output Tax)
2010101002 - CBS Payable (Output Tax)
3010501001 - IBS Expense (Non-recoverable)
3010501002 - CBS Expense (Non-recoverable)
```

#### Configure Automatic Account Assignment (VKOA)

Transaction: **VKOA**

```
Chart of Accounts: YBRA (Brazil)
Condition Type: ZIBS
Account Key: MWS
General Modification:
  - Debit Account: 1010301001 (IBS Recoverable)
  - Credit Account: 2010101001 (IBS Payable)

Condition Type: ZCBS
Account Key: MW2
General Modification:
  - Debit Account: 1010301002 (CBS Recoverable)
  - Credit Account: 2010101002 (CBS Payable)
```

### 4. Material Master Extension

#### Add Custom Fields (Transaction: SE11)

Create table **ZMARA_TAX_REFORM**:

```abap
@EndUserText.label : 'Material - Tax Reform Data'
@AbapCatalog.enhancementCategory : #EXTENSIBLE_ANY
define table zmara_tax_reform {
  key matnr : matnr;         // Material Number
  ncm_code  : char10;        // NCM Classification
  is_essential_good : abap_bool;  // Essential good flag (reduced rate)
  is_health_service : abap_bool;  // Health service flag
  is_education      : abap_bool;  // Education service flag
  ibs_rate_override : dec5_2;     // Manual IBS rate override
  cbs_rate_override : dec5_2;     // Manual CBS rate override
  last_changed_date : dats;       // Last change date
  last_changed_by   : uname;      // Last changed by user
}
```

#### Screen Enhancement (Transaction: SHDO)

Add custom tab in MM01/MM02:
- Tab Name: "Tax Reform"
- Fields: NCM, Essential Good checkbox, Health Service checkbox, etc.

### 5. Procurement (MM) Integration

#### Purchase Order Tax Calculation

**BADI:** ME_PROCESS_PO_CUST
**Method:** PROCESS_ITEM

```abap
METHOD if_ex_me_process_po_cust~process_item.

  DATA: ls_tax_request TYPE zs_tax_request,
        ls_tax_result TYPE zs_tax_result.

  " Get PO item data
  DATA(ls_item) = im_item->get_data( ).

  " Prepare tax calculation request
  ls_tax_request-base_value = ls_item-netwr.
  ls_tax_request-year = sy-datum(4).
  ls_tax_request-ncm = ls_item-ncm_code.
  ls_tax_request-origin_state = 'EX'. "External (import)
  ls_tax_request-destination_state = ls_item-plant_region.
  ls_tax_request-is_export = abap_false.

  " Call tax service
  CALL FUNCTION 'Z_CALCULATE_TAX_API'
    EXPORTING
      is_request = ls_tax_request
    IMPORTING
      es_result  = ls_tax_result
    EXCEPTIONS
      api_error  = 1
      OTHERS     = 2.

  IF sy-subrc = 0.
    " Update PO conditions
    im_item->set_condition( iv_cond_type = 'ZIBS'
                           iv_cond_value = ls_tax_result-ibs_amount ).
    im_item->set_condition( iv_cond_type = 'ZCBS'
                           iv_cond_value = ls_tax_result-cbs_amount ).
  ENDIF.

ENDMETHOD.
```

### 6. Invoice Verification (MIRO) Integration

Automatically validate tax amounts against calculated values:

```abap
" BADI: INVOICE_UPDATE
METHOD if_ex_invoice_update~change_at_save.

  DATA: lt_calculated TYPE TABLE OF zs_tax_line,
        lt_invoice TYPE TABLE OF zs_tax_line.

  " Extract tax lines from invoice
  LOOP AT rbkpv-xblnr ASSIGNING FIELD-SYMBOL(<invoice>).
    " Get ZIBS/ZCBS amounts from invoice
    SELECT kschl, kwert
      FROM bseg
      WHERE belnr = <invoice>-belnr
        AND kschl IN ('ZIBS', 'ZCBS')
      APPENDING TABLE @lt_invoice.
  ENDLOOP.

  " Recalculate taxes via API
  CALL FUNCTION 'Z_RECALCULATE_TAX'
    EXPORTING
      iv_invoice_number = rbkpv-xblnr
    IMPORTING
      et_calculated_tax = lt_calculated.

  " Compare invoice vs. calculated
  LOOP AT lt_invoice ASSIGNING FIELD-SYMBOL(<inv_tax>).
    READ TABLE lt_calculated ASSIGNING FIELD-SYMBOL(<calc_tax>)
      WITH KEY kschl = <inv_tax>-kschl.

    DATA(lv_difference) = abs( <inv_tax>-kwert - <calc_tax>-kwert ).

    " Tolerance: 0.01 (1 cent)
    IF lv_difference > 0.01.
      " Block invoice for review
      MESSAGE e003(ztax) WITH 'Tax amount mismatch:'
                              <inv_tax>-kschl
                              <inv_tax>-kwert
                              <calc_tax>-kwert.
    ENDIF.
  ENDLOOP.

ENDMETHOD.
```

### 7. Financial Accounting (FI) Reporting

#### Create Custom Report: Tax Reform Analysis

**Transaction:** SE38
**Program:** ZTAX_REFORM_ANALYSIS

```abap
REPORT ztax_reform_analysis.

PARAMETERS: p_bukrs TYPE bukrs OBLIGATORY,
            p_gjahr TYPE gjahr OBLIGATORY,
            p_monat TYPE monat OBLIGATORY.

START-OF-SELECTION.

  " Fetch IBS/CBS data
  SELECT bukrs, gjahr, monat,
         SUM( CASE WHEN shkzg = 'S' THEN dmbtr ELSE dmbtr * -1 END ) as ibs_amount
    FROM bseg
    WHERE bukrs = @p_bukrs
      AND gjahr = @p_gjahr
      AND monat = @p_monat
      AND hkont LIKE '20101010%'  " IBS accounts
    GROUP BY bukrs, gjahr, monat
    INTO TABLE @DATA(lt_ibs).

  " Fetch legacy ICMS data
  SELECT bukrs, gjahr, monat,
         SUM( CASE WHEN shkzg = 'S' THEN dmbtr ELSE dmbtr * -1 END ) as icms_amount
    FROM bseg
    WHERE bukrs = @p_bukrs
      AND gjahr = @p_gjahr
      AND monat = @p_monat
      AND hkont LIKE '21010010%'  " ICMS accounts
    GROUP BY bukrs, gjahr, monat
    INTO TABLE @DATA(lt_icms).

  " Display comparison
  WRITE: / 'Tax Reform Analysis', p_gjahr, p_monat.
  WRITE: / '─────────────────────────────────────────────────────'.
  WRITE: / 'IBS (New System):', lt_ibs[ 1 ]-ibs_amount.
  WRITE: / 'ICMS (Legacy):', lt_icms[ 1 ]-icms_amount.
  WRITE: / 'Total Tax Burden:', lt_ibs[ 1 ]-ibs_amount + lt_icms[ 1 ]-icms_amount.
  WRITE: / '─────────────────────────────────────────────────────'.

END-OF-SELECTION.
```

### 8. Testing & Validation

#### Unit Test Checklist

- [ ] IBS calculation in sales order (VA01)
- [ ] CBS calculation in sales order (VA01)
- [ ] Transition period logic (compare 2026 vs. 2033)
- [ ] Export operation exemption
- [ ] Purchase order with IBS/CBS (ME21N)
- [ ] Invoice verification tolerance (MIRO)
- [ ] GL account posting (FB03)
- [ ] Month-end closing with new accounts

#### Integration Test Scenarios

**Test Case 1: Domestic Sale (Intrastate)**
```
Material: 10000001 (Electronics - NCM 85176200)
Quantity: 10 units
Price: R$ 1,000/unit
Origin: SP
Destination: SP (same state)
Year: 2027

Expected:
- IBS: R$ 530.00 (26.5% * 20% transition)
- CBS: R$ 176.00 (8.8% * 20% transition)
- ICMS: R$ 1,440.00 (18% * 80% legacy)
- PIS: R$ 132.00 (1.65% * 80% legacy)
- COFINS: R$ 608.00 (7.6% * 80% legacy)
Total Tax: R$ 2,886.00
```

**Test Case 2: Interstate Sale**
```
Material: 10000002 (Food - NCM 04021010)
Quantity: 100 units
Price: R$ 50/unit
Origin: RS
Destination: SC
Year: 2030

Expected:
- IBS: R$ 397.50 (15.9% * 50% - essential good reduced rate)
- CBS: R$ 132.50 (5.3% * 50%)
- ICMS: R$ 425.00 (17% * 50% legacy)
- PIS: R$ 41.25 (1.65% * 50% legacy)
- COFINS: R$ 190.00 (7.6% * 50% legacy)
Total Tax: R$ 1,186.25
```

## Troubleshooting

### Common Issues

**Issue 1: "Tax API not responding"**
- **Cause:** Network connectivity or API service down
- **Solution:** Check HTTP destination configuration (SM59), verify API health endpoint

**Issue 2: "Tax amount mismatch in MIRO"**
- **Cause:** Vendor invoice uses different tax rates
- **Solution:** Review blocked invoice, validate NCM classification, approve manually if justified

**Issue 3: "Condition type ZIBS not found"**
- **Cause:** Pricing procedure not configured
- **Solution:** Execute configuration steps in section 1 above

## Performance Considerations

- **Cache tax rates:** Store frequently used rates in SAP custom tables (refresh daily)
- **Async API calls:** For batch operations (month-end), use background jobs
- **Fallback mechanism:** Implement table-based calculation if API unavailable

## Security

- **API Authentication:** Use OAuth 2.0 client credentials
- **SAP Authorization:** Create custom auth object Z_TAX_REFORM
- **Encryption:** All API calls via HTTPS only

## Support

For SAP integration issues:
- SAP Consultant: Eduardo Lara Peiter
- Email: dudu.peiter@gmail.com
- GitHub Issues: https://github.com/Dudomon/BrazilianTaxReform-System/issues

---

**Document Version:** 1.0
**Last Updated:** December 2024

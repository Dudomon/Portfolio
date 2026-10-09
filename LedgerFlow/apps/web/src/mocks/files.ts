import { LineItemSchema, type DocumentDetail, type LineItem } from '@/api';

/** Builds a minimal, valid single-page PDF that prints the given lines. */
export function buildPdf(lines: readonly string[]): Uint8Array {
  // Standard 14 fonts only cover Latin-1; strip accents to stay ASCII.
  const ascii = (text: string) =>
    text
      .normalize('NFD')
      .replace(/\p{M}/gu, '')
      .replace(/[^\x20-\x7e]/g, '?')
      .replace(/([\\()])/g, '\\$1');
  const content = [
    'BT',
    '/F1 11 Tf',
    '14 TL',
    '56 780 Td',
    ...lines.map((line) => `(${ascii(line)}) Tj T*`),
    'ET',
  ].join('\n');

  const objects = [
    '<< /Type /Catalog /Pages 2 0 R >>',
    '<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
    '<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>',
    '<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>',
    `<< /Length ${content.length} >>\nstream\n${content}\nendstream`,
  ];

  let pdf = '%PDF-1.4\n';
  const offsets: number[] = [];
  objects.forEach((object, index) => {
    offsets.push(pdf.length);
    pdf += `${index + 1} 0 obj\n${object}\nendobj\n`;
  });
  const xrefOffset = pdf.length;
  pdf += `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n`;
  pdf += offsets.map((offset) => `${offset.toString().padStart(10, '0')} 00000 n \n`).join('');
  pdf += `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xrefOffset}\n%%EOF\n`;
  return new TextEncoder().encode(pdf);
}

function fieldText(document: DocumentDetail, key: string): string {
  const value = document.extraction?.fields[key]?.value;
  return typeof value === 'string' ? value : '';
}

function lineItems(document: DocumentDetail): LineItem[] {
  const parsed = LineItemSchema.array().safeParse(document.extraction?.fields.line_items?.value);
  return parsed.success ? parsed.data : [];
}

export function renderInvoicePdf(document: DocumentDetail): Uint8Array {
  const currency = fieldText(document, 'currency');
  return buildPdf([
    'INVOICE',
    '',
    `Supplier:  ${fieldText(document, 'supplier_name')}`,
    `Tax ID:    ${fieldText(document, 'supplier_tax_id')}`,
    `Number:    ${fieldText(document, 'invoice_number')}`,
    `Issued:    ${fieldText(document, 'issue_date')}`,
    `Due:       ${fieldText(document, 'due_date')}`,
    '',
    'Qty   Unit price   Total        Description',
    ...lineItems(document).map(
      (item) =>
        `${item.quantity.padEnd(6)}${item.unit_price.padEnd(13)}${item.total.padEnd(13)}${item.description}`,
    ),
    '',
    `Tax:       ${fieldText(document, 'tax_amount')} ${currency}`,
    `TOTAL:     ${fieldText(document, 'total_amount')} ${currency}`,
  ]);
}

const escapeXml = (text: string) =>
  text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

export function renderNfeXml(document: DocumentDetail): string {
  const items = lineItems(document)
    .map(
      (item, index) =>
        `    <det nItem="${index + 1}">\n      <prod><xProd>${escapeXml(item.description)}</xProd><qCom>${item.quantity}</qCom><vUnCom>${item.unit_price}</vUnCom><vProd>${item.total}</vProd></prod>\n    </det>`,
    )
    .join('\n');
  return `<?xml version="1.0" encoding="UTF-8"?>
<nfeProc xmlns="http://www.portalfiscal.inf.br/nfe">
  <infNFe Id="NFe${fieldText(document, 'nfe_access_key')}">
    <ide><nNF>${escapeXml(fieldText(document, 'invoice_number'))}</nNF><dhEmi>${fieldText(document, 'issue_date')}</dhEmi></ide>
    <emit><CNPJ>${fieldText(document, 'supplier_tax_id')}</CNPJ><xNome>${escapeXml(fieldText(document, 'supplier_name'))}</xNome></emit>
${items}
    <total><ICMSTot><vICMS>${fieldText(document, 'tax_amount')}</vICMS><vNF>${fieldText(document, 'total_amount')}</vNF></ICMSTot></total>
  </infNFe>
</nfeProc>
`;
}

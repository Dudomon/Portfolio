import type { DocumentStatus } from '@/api';
import { Badge, type BadgeTone } from '@/ui';

export const STATUS_META: Record<DocumentStatus, { label: string; tone: BadgeTone }> = {
  uploaded: { label: 'Uploaded', tone: 'neutral' },
  queued: { label: 'Queued', tone: 'neutral' },
  processing: { label: 'Processing', tone: 'info' },
  extracted: { label: 'Extracted', tone: 'accent' },
  needs_review: { label: 'Needs review', tone: 'warning' },
  approved: { label: 'Approved', tone: 'success' },
  rejected: { label: 'Rejected', tone: 'danger' },
  exported: { label: 'Exported', tone: 'success' },
  failed: { label: 'Failed', tone: 'danger' },
};

export function StatusBadge({ status }: { status: DocumentStatus }) {
  const { label, tone } = STATUS_META[status];
  return (
    <Badge tone={tone} dot>
      {label}
    </Badge>
  );
}
